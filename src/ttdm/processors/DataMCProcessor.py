import awkward as ak
import hist
from coffea import processor
from coffea.analysis_tools import PackedSelection
from collections import OrderedDict
#from coffea.lumi_tools import LumiData, LumiList, LumiMask

REGIONS = ["6j2b", "4j1b"]

# b-tagging working points for Summer24
BTAG_TIGHT = 0.6373
BTAG_MED = 0.1990
BTAG_LOOSE = 0.0365

# Golden JSON for 2024
GOLDEN_JSON_PATH = "data/json/2024I_Golden.json"
LUMIS_2024 = ""


def get_good_jets(jets: ak.Array):
    tight_jetID = (
        (jets.neHEF < 0.99)
        & (jets.neEmEF < 0.9)
        & (jets.nConstituents > 1)
        & (jets.muEF < 0.80)  # For LepVeto
        & (jets.chHEF > 0.01)
        & (jets.nCh > 0)
        & (jets.chEmEF < 0.80)  # For LepVeto
    )
    kinematic_cuts = (jets.pt > 30) & (abs(jets.eta) <= 2.6)
    return jets[tight_jetID & kinematic_cuts]


def get_ea(abs_eta):
    return ak.where(
        abs_eta < 1.0,
        0.1243,
        ak.where(
            abs_eta < 1.479,
            0.1458,
            ak.where(
                abs_eta < 2.0,
                0.0992,
                ak.where(
                    abs_eta < 2.2,
                    0.0794,
                    ak.where(
                        abs_eta < 2.3,
                        0.0762,
                        ak.where(abs_eta < 2.4, 0.0766, ak.where(abs_eta < 2.5, 0.1003, 0.1003)),
                    ),
                ),
            ),
        ),
    )


def get_good_electrons(electrons: ak.Array, rho: ak.Array):
    """
    Electron veto ID computation taken from https://github.com/tihsu99/EGM_Scouting_analysis/tree/main
    """
    veto_map = {
        "sieie": 0.0117,
        "dEta": 0.0071,
        "dPhi": 0.208,
        "H/E": lambda E, p: 0.05 + 1.28 / E + 0.0422 * p / E,
        "iso": lambda pt: 0.406 + 0.535 / pt,
        "ooEmooP": 0.178,
        "missHits": 2,
    }

    abs_eta = abs(electrons.eta)
    is_barrel = abs_eta <= 1.479
    pt = electrons.pt
    # esc = electrons.rawEnergy
    rho = rho
    effective_area = get_ea(abs_eta)
    sieie = electrons.sigmaIetaIeta
    dEta = abs(electrons.dEtaIn)
    dPhi = abs(electrons.dPhiIn)
    # hOverE = electrons.hOverE

    neutral_iso = electrons.ecalIso + electrons.hcalIso - rho * effective_area
    iso = (ak.where(neutral_iso > 0, neutral_iso, 0) + electrons.trackIso) / pt
    ooEmooP = abs(electrons.ooEMOop)
    missingHits = electrons.missingHits

    barrel_mask = (
        is_barrel
        & (sieie < veto_map["sieie"])
        & (dEta < veto_map["dEta"])
        & (dPhi < veto_map["dPhi"])
        & (iso < veto_map["iso"](pt))
        & (ooEmooP < veto_map["ooEmooP"])
        & (missingHits <= veto_map["missHits"])
        # & (hOverE < veto_map["H/E"](esc, rho)) # how do I get electron energy
    )

    return electrons[barrel_mask]


def get_good_muons(muons: ak.Array):
    veto_muonID = (
        (muons.normchi2 < 10.0)
        & (muons.nValidRecoMuonHits > 0)
        & (muons.nRecoMuonMatchedStations > 1)
        & (abs(muons.trk_dxy) < 0.5)
        & (abs(muons.trk_dz) < 0.5)
        & (muons.nValidPixelHits > 0)
        & (muons.nTrackerLayersWithMeasurement > 5)
    )  # Using tight muon ID requirements
    kinematic_cuts = (muons.pt > 3) & (abs(muons.eta) < 2.4)

    return muons[veto_muonID & kinematic_cuts]


def apply_triggers(events) -> ak.Array:
    triggers = PackedSelection()
    triggers.add("L1_AXO_Nominal", events.L1.AXO_Nominal)
    triggers.add("L1_HTT360er", events.L1.HTT360er)
    triggers.add("L1_ETMHF70", events.L1.ETMHF70)
    return triggers.any(*triggers.names)


def region_selection(jets, electrons, muons) -> PackedSelection:
    """Store boolean masks with PackedSelection"""
    selections = PackedSelection(dtype="uint64")

    selections.add("exactly_4j", ak.num(jets, axis=1) == 4)
    selections.add("atleast_6j", ak.num(jets, axis=1) >= 6)
    selections.add("exactly_1b", ak.sum(jets.particleNet_prob_b >= BTAG_LOOSE, axis=1) == 1)
    selections.add("atleast_2b", ak.sum(jets.particleNet_prob_b >= BTAG_LOOSE, axis=1) >= 2)
    selections.add("no_leptons", (ak.num(electrons) + ak.num(muons)) == 0)
    # Define control and signal region
    selections.add("6j2b", selections.all("no_leptons", "atleast_6j", "atleast_2b"))
    selections.add("4j1b", selections.all("no_leptons", "exactly_4j", "exactly_1b"))

    return selections


class DataMCProcessor(processor.ProcessorABC):
    """
    Perform initial Data-MC comparisons by
    * Applying the golden JSON via `LumiMask`
    * Storing the raw number of events processed for MC, or total luminosity run over for data
    """

    def __init__(self, isMC: bool = False, era: str = "2024", mode: str = "delayed"):
        """
        Parameters
        ----------
            isMC: bool
                If True, the passed data is Monte Carlo. Setting this parameter to True
                disables luminosity calculations, and instead tracks the raw number of
                events processed by the processor for cross-section scaling.
            era: str
                The era passed data belongs to.
        """
        super().__init__()
        self.isMC = isMC
        self.era = era
        self.cutflow = OrderedDict()

        # Define axes for histograms
        self.met_pt_axis = hist.axis.Regular(
            70, 0, 700, name="met", label=r"$E^{\rm miss}_T$ [GeV]", underflow=False, overflow=False
        )
        self.event_ht_axis = hist.axis.Regular(
            200, 0, 2000, name="ht", label=r"$H_T$ [GeV]", underflow=False, overflow=False
        )
        self.pt_axis = hist.axis.Regular(
            150, 0, 1500, name="pt", label=r"$p_T$ [GeV]", underflow=False, overflow=False
        )
        self.multiplicity_axis = hist.axis.Regular(
            50, 0, 50, name="njets", label="Number of Jets", underflow=False, overflow=False
        )
        self.region_axis = hist.axis.StrCategory(
            REGIONS, growth=False, name="region", label="Selection Region"
        )

        # Define output histogram

        if mode == "delayed":
            self.output = {
                "njet": hist.dask.Hist(
                    self.multiplicity_axis, self.region_axis, storage=hist.storage.Weight()
                ),
                "pt": hist.dask.Hist(self.pt_axis, self.region_axis, storage=hist.storage.Weight()),
                "met": hist.dask.Hist(self.met_pt_axis, self.region_axis, storage=hist.storage.Weight()),
                "ht": hist.dask.Hist(self.event_ht_axis, self.region_axis, storage=hist.storage.Weight()),
            }
        else:
            self.output = {
                "njet": hist.Hist(
                    self.multiplicity_axis, self.region_axis, storage=hist.storage.Weight()
                ),
                "pt": hist.Hist(self.pt_axis, self.region_axis, storage=hist.storage.Weight()),
                "met": hist.Hist(self.met_pt_axis, self.region_axis, storage=hist.storage.Weight()),
                "ht": hist.Hist(self.event_ht_axis, self.region_axis, storage=hist.storage.Weight()),
            }

        if self.isMC:
            counter_type = "nevents"
        else:
            counter_type = "luminosity"

        self.output[counter_type] = 0

    def process(self, events) -> dict[str, hist.Hist]:
        if self.isMC:
            self.output["nevents"] += ak.num(events.ScoutingPFJetRecluster, axis=0)
            self.cutflow["begin_jets"] = ak.num(events.ScoutingPFJetRecluster, axis=1)
        else:
            pass
            # FIXME: coffea v2025.7 images required for this to work
            # year_lumi_mask = LumiMask(GOLDEN_JSON_PATH)
            # events = events[year_lumi_mask(runs=events.run, lumis=events.luminosityBlock)]
            # lumi_list = LumiList(runs=events.run, lumis=events.luminosityBlock)
            # self.output["luminosity"] += LumiData(LUMIS_2024).get_lumi(lumi_list)

        # Apply object selection, triggers, and offline event selections
        good_jets = get_good_jets(events.ScoutingPFJetRecluster)
        good_electrons, good_muons = get_good_electrons(events.ScoutingElectron, rho=events.ScoutingRho), get_good_muons(
            events.ScoutingMuonVtx
        )
        self.cutflow["jetid"] = ak.num(good_jets, axis=1)
        trigger_filter = apply_triggers(events)

        jets = good_jets[trigger_filter]
        veto_electrons = good_electrons[trigger_filter]
        veto_muons = good_muons[trigger_filter]
        self.cutflow["trigger"] = ak.num(jets, axis=1)

        # region selections
        selections = region_selection(jets, veto_electrons, veto_muons)

        for region in REGIONS:
            selection = selections.all(region)
            region_jets = jets[selection]
            region_met = events.ScoutingMET[selection]
            self.cutflow[region] = ak.num(region_jets, axis=1)
            self.output["njet"].fill(njets=ak.num(region_jets, axis=1), region=region)
            self.output["pt"].fill(pt=ak.flatten(region_jets.pt, axis=1), region=region)
            self.output["ht"].fill(ht=ak.sum(region_jets.pt, axis=1), region=region)
            self.output["met"].fill(met=region_met.pt, region=region)
            self.output["cutflow"] = self.cutflow
        return self.output

    def postprocess(self, accumulator):
        return super().postprocess(accumulator)
