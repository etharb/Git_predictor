# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class HRMHead(nn.Module):
    """
    HRM-lite: dvouúrovňová rekurence pro časové řady.
    - high GRU (pomalejší plán): aktualizuje se po každých high_period krocích
    - low GRU (rychlý detail): na každém kroku dostává [x_t || h_high] a generuje finální stav
    - head_updown: MLP -> sigmoid (zpětně kompatibilní výstup p_up)
    - volitelné pomocné hlavy:
        * head_abstain: sigmoid (p_abstain)
        * head_softmax3: softmax nad třídami [DOWN, ABSTAIN, UP]
        * head_uncert: sigmoid (p_ne jistoty) – nepovinné, typicky kopíruje ABSTAIN
        * head_hsel: softmax přes N hlav/horizontů (pokud N>0)

    Podporované vstupy:
      - sekvenční: x.shape == (B, T, F)
      - jednorázový „vektor“ (L1 meta v sadě): x.shape == (B, F)  -> interně se přetvoří na T=1

    Pozn.: forward() vrací POUZE p_up (tensor (B,)), aby zůstala kompatibilita s existujícím kódem.
           Pro trénování vícemodových hlav použij forward_heads()/infer_heads(), které vrací dict s klíči.
    """
    def __init__(
        self,
        in_features: int,
        hidden_low: int = 64,
        hidden_high: int = 64,
        high_period: int = 10,
        use_abstain_head: bool = True,
        use_softmax3: bool = False,
        use_uncert_head: bool = False,
        num_hsel: int = 0
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_low = int(hidden_low)
        self.hidden_high = int(hidden_high)
        self.high_period = max(1, int(high_period))

        # GRU bloky
        self.high = nn.GRU(input_size=self.in_features, hidden_size=self.hidden_high, batch_first=True)
        self.low  = nn.GRU(input_size=self.in_features + self.hidden_high, hidden_size=self.hidden_low, batch_first=True)

        # Hlavní binární hlava: P(UP)
        self.head_updown = nn.Sequential(
            nn.Linear(self.hidden_low, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Volitelné pomocné hlavy
        self.use_abstain_head = bool(use_abstain_head)
        self.use_softmax3 = bool(use_softmax3)
        self.use_uncert_head = bool(use_uncert_head)
        self.num_hsel = int(num_hsel)

        if self.use_abstain_head:
            self.head_abstain = nn.Sequential(
                nn.Linear(self.hidden_low, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.head_abstain = None

        if self.use_softmax3:
            self.head_softmax3 = nn.Sequential(
                nn.Linear(self.hidden_low, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
                nn.Softmax(dim=-1)
            )
        else:
            self.head_softmax3 = None

        if self.use_uncert_head:
            self.head_uncert = nn.Sequential(
                nn.Linear(self.hidden_low, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.head_uncert = None

        if self.num_hsel and self.num_hsel > 1:
            self.head_hsel = nn.Sequential(
                nn.Linear(self.hidden_low, max(32, self.num_hsel)),
                nn.ReLU(inplace=True),
                nn.Linear(max(32, self.num_hsel), self.num_hsel),
                nn.Softmax(dim=-1)
            )
        else:
            self.head_hsel = None

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Eval-only helper. Vrací (B,) pravděpodobnost P(UP).
        """
        self.eval()
        return self.forward(x)

    @torch.no_grad()
    def infer_heads(self, x: torch.Tensor) -> dict:
        """
        Eval-only helper. Vrací slovník se všemi dostupnými hlavami.
        Klíče: 'updown' (B,), volitelně 'abstain' (B,), 'softmax3' (B,3), 'uncert' (B,), 'hsel' (B,num_hsel)
        """
        self.eval()
        return self.forward_heads(x)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Společná část: GRU průchod a návrat finálního low-level stavu h_low (B,Hl).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        assert x.dim() == 3, f"HRMHead: očekávám (B,T,F) nebo (B,F), dostal jsem shape={tuple(x.shape)}"
        B, T, F = x.shape
        assert F == self.in_features, f"HRMHead: F={F} neodpovídá in_features={self.in_features}"

        device = x.device
        h_high = torch.zeros(1, B, self.hidden_high, device=device)
        h_low  = torch.zeros(1, B, self.hidden_low,  device=device)

        for t in range(T):
            x_t = x[:, t:t+1, :]
            if (t % self.high_period) == 0:
                _, h_high = self.high(x_t, h_high)
            high_ctx = h_high.permute(1, 0, 2)  # (B,1,Hh)
            low_in = torch.cat([x_t, high_ctx], dim=-1)  # (B,1,F+Hh)
            _, h_low = self.low(low_in, h_low)

        return h_low.squeeze(0)  # (B,Hl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zpětně kompatibilní výstup: jen p_up (B,).
        """
        h = self._encode(x)
        p_up = self.head_updown(h).squeeze(-1)
        return p_up

    def forward_heads(self, x: torch.Tensor) -> dict:
        """
        Kompletní multi-head výstup.
        """
        h = self._encode(x)
        out = {
            "updown": self.head_updown(h).squeeze(-1)
        }
        if self.head_abstain is not None:
            out["abstain"] = self.head_abstain(h).squeeze(-1)
        if self.head_softmax3 is not None:
            out["softmax3"] = self.head_softmax3(h)
        if self.head_uncert is not None:
            out["uncert"] = self.head_uncert(h).squeeze(-1)
        if self.head_hsel is not None:
            out["hsel"] = self.head_hsel(h)
        return out