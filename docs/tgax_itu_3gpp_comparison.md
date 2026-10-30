# TGax outdoor versus the existing 3GPP implementation

This decision record compares IEEE 802.11-14/0882r4's outdoor baseline
(ITU-R M.2135-1 UMi/UMa) with this repository's TR 38.901 V16.1.0
implementation. “Reusable” means algorithmic infrastructure may be shared;
it never changes the TGax source identity.

| Field | IEEE TGax / ITU-R baseline | Existing TR 38.901 V16.1 | Decision |
| --- | --- | --- | --- |
| Normative identity | 14/0882r4 referencing M.2135-1 | TR 38.901 V16.1.0 | Incompatible; separate resources and metadata |
| Coordinates and 3-D distance | Cartesian AP/STA geometry | Cartesian BS/UT geometry | Identical mathematics; reuse topology helpers |
| Carrier range | 2-6 GHz stated for ITU outdoor SCM | Version-specific range extends beyond 6 GHz | Similar only; TGax validates its own range |
| UMi LoS probability | 18-m, 36-m expression | Similar UMi expression | Mathematically similar; retain IEEE-owned resource |
| UMa LoS probability | 18-m, 63-m expression | Height-corrected TR 38.901 expression | Incompatible |
| UMi LoS path loss | 22 dB/decade before a height breakpoint; 40 after | Related but versioned formula/applicability | IEEE-specific implementation |
| UMi NLoS path loss | `22.7 + 36.7 log10(d) + 26 log10(fc_GHz)` | Different coefficients | Incompatible |
| UMa NLoS path loss | M.2135 street-width/building-height expression | TR 38.901 UMa expression | Similar family, different calibration; IEEE-specific implementation |
| Shadowing | 3 dB LoS, 4 dB NLoS, 7 dB O2I | Scenario/state-specific TR table | Separate resource |
| O2I | NLoS outdoor loss plus `20 + 0.5*d_in` | Material/frequency-dependent model | Incompatible |
| Delay spreads | TGax table II: UMi 65/129/240 ns; UMa 93/363 ns | Lognormal LSP tables | Incompatible calibration |
| LSP correlation | ITU-R calibration | TR 38.901 correlation tables | Not reused |
| Cluster count/rays | ITU-R SCM | TR 38.901 scenario tables | Not reused |
| Spatial consistency | ITU-R/system-evaluation assumptions | TR 38.901 procedure | Not reused without proof |
| Fast objects | clusters 2 and 3 at 60 km/h; remainder 3 km/h | Different construction | IEEE-specific overlay |
| Backend, RNG, PSD roots, CIR/OFDM contracts | Framework concerns | Same framework concerns | Reusable without sharing physics |

The claimed outdoor TGax subset includes source-specific path loss, shadowing,
topology applicability, and fixed-window fast-object behavior. Full ITU-R
outdoor LSP/cluster calibration remains outside that claim; the API makes the
selected A-F small-scale kernel explicit and documentation must not describe
it as a complete M.2135 spatial channel realization.
