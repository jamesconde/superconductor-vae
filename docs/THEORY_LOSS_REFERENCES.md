# Theory Loss References

Citations for the physics formulas and empirical constraints used in
`src/superconductor/losses/theory_losses.py`. Organized by loss class.

---

## BCS / Conventional Superconductors (`BCSTheoryLoss`)

### Foundational Theory

[BCS1957] J. Bardeen, L. N. Cooper, and J. R. Schrieffer, "Theory of Superconductivity,"
*Physical Review* **108**, 1175--1204 (1957).
DOI: 10.1103/PhysRev.108.1175

### McMillan Formula (superseded by Allen-Dynes in V12.25)

[McMillan1968] W. L. McMillan, "Transition Temperature of Strong-Coupled Superconductors,"
*Physical Review* **167**, 331--344 (1968).
DOI: 10.1103/PhysRev.167.331

> Original Tc formula: Tc = (theta_D / 1.45) * exp(-1.04(1+lambda)/(lambda - mu*(1+0.62*lambda))).
> Accurate to ~20-30% for weak-coupling (lambda < 0.5). Systematically underestimates
> Tc for strong-coupling materials (lambda > 0.5) like MgB2 or Pb.

### Allen-Dynes Formula (used in V12.25)

[AllenDynes1975] P. B. Allen and R. C. Dynes, "Transition temperature of strong-coupled
superconductors reanalyzed," *Physical Review B* **12**, 905--922 (1975).
DOI: 10.1103/PhysRevB.12.905

> Strong-coupling extension of McMillan. Adds correction factors f1, f2 that account
> for the shape of the phonon spectrum and anharmonic effects. Recovers McMillan
> in the weak-coupling limit (f1, f2 -> 1). The omega_log ~ 0.827 * theta_D
> approximation is for monatomic/simple polyatomic solids.

### Lindemann Debye Temperature Anchor

[Lindemann1910] F. A. Lindemann, "Uber die Berechnung molekularer Eigenfrequenzen,"
*Physikalische Zeitschrift* **11**, 609--612 (1910).

[Grimvall1999] G. Grimvall, *Thermophysical Properties of Materials*, Enlarged and
Revised Edition (Elsevier/North-Holland, Amsterdam, 1999).
ISBN: 978-0-444-82794-4.

> Lindemann (1910) proposed that melting occurs when atomic vibration amplitude
> reaches a critical fraction of the lattice spacing. Grimvall (1999, Part 3)
> derives the connection: theta_D ~ C * sqrt(T_m / (M * V^(2/3))), where T_m is
> the melting temperature, M is mean atomic weight, and V is volume per atom.
> We use C = 41.63 (in units giving theta_D in Kelvin when T_m is in Kelvin,
> M in AMU, V in angstrom^3). This is a soft anchor on the neural net theta_D
> predictor, not a replacement.

### Matthias VEC Rule

[Matthias1955] B. T. Matthias, "Empirical Relation between Superconductivity and the
Number of Valence Electrons per Atom," *Physical Review* **97**, 74--76 (1955).
DOI: 10.1103/PhysRev.97.74

> Matthias observed that Tc in transition metal alloys peaks near valence electron
> count (VEC) ~ 4.7 and ~ 6.7. This empirical correlation has held for decades
> of subsequent data. We implement it as a very gentle (weight 0.01) Gaussian
> envelope prior on BCS Tc predictions.

---

## Cuprate Superconductors (`CuprateTheoryLoss`)

### Presland Doping Dome

[Presland1991] M. R. Presland, J. L. Tallon, R. G. Buckley, R. S. Liu, and N. E. Flower,
"General trends in oxygen stoichiometry effects on Tc in Bi and Tl superconductors,"
*Physica C* **176**, 95--105 (1991).
DOI: 10.1016/0921-4534(91)90700-9

> Universal parabolic dome: Tc/Tc_max = 1 - 82.6*(p - 0.16)^2, where p is the
> hole doping level and 0.16 is optimal doping. This relationship has been
> validated across all major cuprate families (YBCO, LSCO, Bi-2212, Tl-2223,
> Hg-1223). We use learnable predictors for both p and Tc_max from Magpie features.

---

## Iron-Based Superconductors (`IronBasedTheoryLoss`)

### Discovery

[Kamihara2008] Y. Kamihara, T. Watanabe, M. Hirano, and H. Hosono, "Iron-Based Layered
Superconductor La[O1-xFx]FeAs (x = 0.05-0.12) with Tc = 26 K," *Journal of the
American Chemical Society* **130**, 3296--3297 (2008).
DOI: 10.1021/ja800073m

### Comprehensive Review

[Stewart2011] G. R. Stewart, "Superconductivity in iron compounds," *Reviews of Modern
Physics* **83**, 1589--1652 (2011).
DOI: 10.1103/RevModPhys.83.1589. arXiv: 1106.1618.

> 64-page review covering all known iron-based SC families (1111, 122, 11, 111, etc.)
> and their properties. Highest Tc in iron-based family: SmFeAsO1-xFx at ~55K.
> Soft cap at 60K covers all known members with margin.

### VEC Constraint

[Hosono2015] H. Hosono and K. Kuroki, "Iron-based superconductors: Current status of
materials and pairing mechanism," *Physica C* **514**, 399--422 (2015).
DOI: 10.1016/j.physc.2015.02.020. arXiv: 1504.04919.

> Discusses how parent iron pnictides have Fe2+ with 6 d-electrons (VEC = 6 per Fe site)
> and how doping away from this electron count affects Tc. The d6 configuration
> places the system at half-filling of the t2g manifold, optimizing nesting and
> magnetic fluctuation-driven pairing.

---

## Heavy Fermion Superconductors (`HeavyFermionTheoryLoss`)

### Discovery (First Heavy Fermion SC)

[Steglich1979] F. Steglich, J. Aarts, C. D. Bredl, W. Lieke, D. Meschede, W. Franz,
and H. Schafer, "Superconductivity in the Presence of Strong Pauli Paramagnetism:
CeCu2Si2," *Physical Review Letters* **43**, 1892--1896 (1979).
DOI: 10.1103/PhysRevLett.43.1892

> CeCu2Si2 with Tc ~ 0.5K was the first heavy fermion superconductor. Established
> that superconductivity can coexist with (and emerge from) strongly correlated
> electron states.

### Highest-Tc Heavy Fermion (PuCoGa5)

[Sarrao2002] J. L. Sarrao, L. A. Morales, J. D. Thompson, B. L. Scott, G. R. Stewart,
F. Wastin, J. Rebizant, P. Boulet, E. Colineau, and G. H. Lander, "Plutonium-based
superconductivity with a transition temperature above 18 K," *Nature* **420**, 297--299
(2002).
DOI: 10.1038/nature01212

> PuCoGa5 at Tc = 18.5K is the highest-Tc heavy fermion SC known. Our soft cap
> of 20K is set just above this record.

### Comprehensive Review

[Pfleiderer2009] C. Pfleiderer, "Superconducting phases of f-electron compounds,"
*Reviews of Modern Physics* **81**, 1551--1624 (2009).
DOI: 10.1103/RevModPhys.81.1551

> 74-page review of all Ce, U, Yb, and Pu heavy fermion superconductors.
> Typical Tc range: 0.1-2K. Key materials: CeCoIn5 (2.3K), UPt3 (0.5K),
> CeRhIn5 (2.1K under pressure), UBe13 (0.9K). The log-normal prior
> centered at 1K reflects this distribution.

---

## Organic Superconductors (`OrganicTheoryLoss`)

### First Organic SC

[Jerome1980] D. Jerome, A. Mazaud, M. Ribault, and K. Bechgaard, "Superconductivity
in a synthetic organic conductor (TMTSF)2PF6," *Journal de Physique Lettres* **41**,
L95--L98 (1980).
DOI: 10.1051/jphyslet:0198000410409500

> First organic superconductor at Tc = 0.9K under 12 kbar pressure.
> Established that purely organic conductors can become superconducting.

### First Fullerene SC

[Hebard1991] A. F. Hebard, M. J. Rosseinsky, R. C. Haddon, D. W. Murphy, S. H. Glarum,
T. T. M. Palstra, A. P. Ramirez, and A. R. Kortan, "Superconductivity at 18 K in
potassium-doped C60," *Nature* **350**, 600--601 (1991).
DOI: 10.1038/350600a0

> K3C60 at Tc = 18K. Alkali-doped fullerenes form a distinct sub-family of organic
> superconductors with much higher Tc than BEDT-TTF salts.

### Highest-Tc Fullerene (Cs3C60)

[Ganin2008] A. Y. Ganin, Y. Takabayashi, Y. Z. Khimyak, S. Margadonna, A. Tamai,
M. J. Rosseinsky, and K. Prassides, "Bulk superconductivity at 38 K in a molecular
system," *Nature Materials* **7**, 367--371 (2008).
DOI: 10.1038/nmat2179

[Ganin2010] A. Y. Ganin, Y. Takabayashi, P. Jeglic, D. Arcon, A. Potocnik,
P. J. Baker, Y. Ohishi, M. T. McDonald, M. D. Tzirakis, A. McLennan,
G. R. Darling, M. Takata, M. J. Rosseinsky, and K. Prassides, "Polymorphism
control of superconductivity and magnetism in Cs3C60 close to the Mott transition,"
*Nature* **466**, 221--225 (2010).
DOI: 10.1038/nature09120

> Cs3C60 at Tc = 38K under pressure — the highest Tc among all molecular/organic
> superconductors. Our fullerene cap of 45K and conservative organic cap of 15K
> bracket the two sub-families.

---

## How Citations Map to Code

| Code Location | Formula/Constraint | References |
|---|---|---|
| `BCSTheoryLoss.allen_dynes_tc()` | Allen-Dynes Tc formula | [AllenDynes1975], [McMillan1968], [BCS1957] |
| `BCSTheoryLoss.forward()` (Lindemann anchor) | theta_D ~ sqrt(T_m/(M*V^(2/3))) | [Lindemann1910], [Grimvall1999] |
| `BCSTheoryLoss.forward()` (Matthias VEC) | Tc peaks at VEC=4.7, 6.7 | [Matthias1955] |
| `CuprateTheoryLoss.dome_function()` | Presland parabolic dome | [Presland1991] |
| `IronBasedTheoryLoss.forward()` (Tc cap 60K) | Iron-based Tc upper bound | [Stewart2011], [Kamihara2008] |
| `IronBasedTheoryLoss.forward()` (VEC=6.0) | Fe2+ tetrahedral VEC optimum | [Hosono2015] |
| `HeavyFermionTheoryLoss.forward()` (prior at 1K) | Log-normal Tc distribution | [Pfleiderer2009], [Steglich1979] |
| `HeavyFermionTheoryLoss.forward()` (cap at 20K) | PuCoGa5 record Tc | [Sarrao2002] |
| `OrganicTheoryLoss.forward()` (cap at 15K) | BEDT-TTF Tc range | [Jerome1980] |
| Config: `organic_fullerene_tc_max = 45K` | Cs3C60 record Tc | [Ganin2008], [Ganin2010], [Hebard1991] |
