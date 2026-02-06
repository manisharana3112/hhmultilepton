# coding: utf-8
from columnflow.selection import selector, SelectionResult
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")


@selector(
    uses={
        "GenPart.{pdgId,genPartIdxMother,statusFlags,pt,eta,phi,mass}",
        "Electron.genPartIdx",
        "Muon.genPartIdx",
        "Tau.genPartIdx",
    },
    produces={
        "lep_isPrompt",
        "lep_isFromHiggs",
        "lep_isFromTau",
        "lep_isUnmatched",
        "nFromDiHiggs",
        "mHH_gen",
    },
    exposed=False,
)
def gen_dihiggs_selector(self, events, lepton_results=None, **kwargs):

    gen = events.GenPart

    # ---------------------------------------------------------
    # Helper: Trace ancestry
    # ---------------------------------------------------------
    def get_origin_flags(gen_indices):

        is_prompt = ak.zeros_like(gen_indices, dtype=bool)
        is_from_higgs = ak.zeros_like(gen_indices, dtype=bool)
        is_from_tau = ak.zeros_like(gen_indices, dtype=bool)

        current_indices = ak.mask(gen_indices, gen_indices >= 0)

        for _ in range(15):

            valid = current_indices >= 0
            safe = ak.mask(current_indices, valid)

            parts = gen[safe]

            pdg = ak.fill_none(abs(parts.pdgId), 0)
            flags = ak.fill_none(parts.statusFlags, 0)

            # Prompt bits (0 and 8)
            is_prompt = is_prompt | (
                ((flags & 1) > 0) | ((flags & (1 << 8)) > 0)
            )

            is_from_higgs = is_from_higgs | (pdg == 25)
            is_from_tau = is_from_tau | (pdg == 15)

            current_indices = ak.fill_none(parts.genPartIdxMother, -1)

        return is_prompt, is_from_higgs, is_from_tau


    # ---------------------------------------------------------
    # 1. Per-lepton origin flags
    # ---------------------------------------------------------
    ele_isPrompt, ele_isFromHiggs, ele_isFromTau = get_origin_flags(events.Electron.genPartIdx)
    mu_isPrompt, mu_isFromHiggs, mu_isFromTau = get_origin_flags(events.Muon.genPartIdx)
    tau_isPrompt, tau_isFromHiggs, tau_isFromTau = get_origin_flags(events.Tau.genPartIdx)

    # ---------------------------------------------------------
    # 2. Concatenate per-lepton arrays
    # ---------------------------------------------------------
    lep_isPrompt = ak.concatenate([ele_isPrompt, mu_isPrompt, tau_isPrompt], axis=1)

    lep_isFromHiggs = ak.concatenate(
        [ele_isFromHiggs, mu_isFromHiggs, tau_isFromHiggs],
        axis=1,
    )

    lep_isFromTau = ak.concatenate(
        [ele_isFromTau, mu_isFromTau, tau_isFromTau],
        axis=1,
    )

    lep_isUnmatched = ak.concatenate(
        [
            events.Electron.genPartIdx < 0,
            events.Muon.genPartIdx < 0,
            events.Tau.genPartIdx < 0,
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # 3. Di-Higgs reconstruction (simple & stable)
    # ---------------------------------------------------------

    higgses = gen[abs(gen.pdgId) == 25]
    has_dihiggs = ak.num(higgses, axis=1) >= 2

    # pad to 2 Higgs
    h_pairs = ak.pad_none(higgses, 2)

    # compute mHH
    mHH_gen = (h_pairs[:, 0] + h_pairs[:, 1]).mass
    mHH_gen = ak.fill_none(mHH_gen, -999.0)

    # count leptons from Di-Higgs
    lep_isFromDiHiggs = lep_isFromHiggs & has_dihiggs
    nFromDiHiggs = ak.sum(lep_isFromDiHiggs, axis=1)

    # ---------------------------------------------------------
    # 4. Save columns
    # ---------------------------------------------------------

    events = set_ak_column(events, "lep_isPrompt", lep_isPrompt)
    events = set_ak_column(events, "lep_isFromHiggs", lep_isFromHiggs)
    events = set_ak_column(events, "lep_isFromTau", lep_isFromTau)
    events = set_ak_column(events, "lep_isUnmatched", lep_isUnmatched)

    events = set_ak_column(events, "nFromDiHiggs", nFromDiHiggs)
    events = set_ak_column(events, "mHH_gen", mHH_gen)

    return events, SelectionResult(steps={}, objects={}, aux={})
