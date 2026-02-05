# coding: utf-8
from columnflow.selection import selector, SelectionResult
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
import law

ak = maybe_import("awkward")
np = maybe_import("numpy")
vector = maybe_import("vector")

@selector(
    uses={
        "GenPart.{pdgId,genPartIdxMother,statusFlags,pt,eta,phi,mass}",
        "Electron.genPartIdx",
        "Muon.genPartIdx",
        "Tau.genPartIdx",
    },
    produces={
        "lep_isPrompt",
        "lep_isUnmatched",
        "nFromDiHiggs",
        "mHH_gen",
    },
    exposed=False,
)
def gen_dihiggs_selector(self, events, lepton_results, **kwargs):
    """
    Vectorized Gen-Level Selector.
    Calculates Lepton Origin flags and mHH_gen using vectorized operations.
    If 'lepton_results' is provided, it calculates Event-Level flags based on the selected leptons.
    """
    gen = events.GenPart
    
    # helper for origin flags
    def get_origin_flags(gen_indices):
        # Result arrays initialized to False
        is_prompt = ak.zeros_like(gen_indices, dtype=bool)
        is_from_higgs = ak.zeros_like(gen_indices, dtype=bool)
        is_from_tau = ak.zeros_like(gen_indices, dtype=bool)
        
        # Mask for valid gen matches (indices >= 0)
        has_match = gen_indices >= 0
        
        # --- Prompt Check ---
        valid_indices = ak.mask(gen_indices, has_match)
        matched_gen = gen[valid_indices]
        
        # Check status flags (Bit 0=Prompt, Bit 8=PromptTauDecayProduct)
        flags = ak.fill_none(matched_gen.statusFlags, 0)
        is_prompt = (flags & 1) | (flags & (1 << 8)) > 0
        
        # --- Ancestry Tracing ---
        current_indices = valid_indices 
        for _ in range(15):
            current_parts = gen[current_indices]
            pdgs = ak.fill_none(abs(current_parts.pdgId), 0)
            
            is_from_higgs = is_from_higgs | (pdgs == 25)
            is_from_tau = is_from_tau | (pdgs == 15)
            
            parents = current_parts.genPartIdxMother
            current_indices = parents 

        return is_prompt, is_from_higgs, is_from_tau

    # 1. Calculate Per-Flavor Flags
    # We do this separately now to facilitate event-level checks later
    ele_isPrompt, ele_isFromHiggs, ele_isFromTau = get_origin_flags(events.Electron.genPartIdx)
    mu_isPrompt, mu_isFromHiggs, mu_isFromTau = get_origin_flags(events.Muon.genPartIdx)
    tau_isPrompt, tau_isFromHiggs, tau_isFromTau = get_origin_flags(events.Tau.genPartIdx)
    
    # 2. Reconstruct the concatenated columns (Backward Compatibility / Original Output)
    # Order: Electron -> Muon -> Tau
    lep_isPrompt = ak.concatenate([ele_isPrompt, mu_isPrompt, tau_isPrompt], axis=1)
    lep_isUnmatched = ak.concatenate([events.Electron.genPartIdx < 0, events.Muon.genPartIdx < 0, events.Tau.genPartIdx < 0], axis=1)

    if lepton_results:
        # Get selected indices
        sel_ele = lepton_results.objects.Electron.Electron
        sel_mu = lepton_results.objects.Muon.Muon
        sel_tau = lepton_results.objects.Tau.Tau
        
        # Check selected leptons only
        # We select the flags corresponding to the selected indices
        sel_ele_prompt = ele_isPrompt[sel_ele]
        sel_mu_prompt = mu_isPrompt[sel_mu]
        sel_tau_prompt = tau_isPrompt[sel_tau]
        
        # Event is prompt if ALL selected leptons are prompt
        # We assume empty selection (e.g. 0 electrons) is True (vacuously true), 
        # but combined with other channels it works out.
        all_ele_ok = ak.all(sel_ele_prompt, axis=1)
        all_mu_ok = ak.all(sel_mu_prompt, axis=1)
        all_tau_ok = ak.all(sel_tau_prompt, axis=1)
        
        event_is_prompt = all_ele_ok & all_mu_ok & all_tau_ok
        
        # Unmatched: If ANY selected is unmatched
        # Actually easier: Unmatched = NOT Prompt (simplification, but assuming fake=unmatched+non-prompt)
        # Or strictly unmatched?
        # User asked for "lep_isUnmatched".
        ele_isUn = events.Electron.genPartIdx < 0
        mu_isUn = events.Muon.genPartIdx < 0
        tau_isUn = events.Tau.genPartIdx < 0
        
        sel_ele_un = ele_isUn[sel_ele]
        sel_mu_un = mu_isUn[sel_mu]
        sel_tau_un = tau_isUn[sel_tau]
        
        event_is_unmatched = ak.any(sel_ele_un, axis=1) | ak.any(sel_mu_un, axis=1) | ak.any(sel_tau_un, axis=1)
        
        # Overwrite the output variables with Event-Level BOLEANS (as requested by variable definition)
        lep_isPrompt = event_is_prompt
        lep_isUnmatched = event_is_unmatched

    # 4. Di-Higgs Reconstruction (Leptons -> Higgs -> HH)
    # This logic identifies all final products (leptons, neutrinos, quarks) 
    # that originate from Higgs bosons (PDG 25).
    gen_idx = ak.local_index(gen)
    _, p_is_from_h, _ = get_origin_flags(gen_idx)
    
    # Terminal products: isLastCopy (bit 13)
    is_terminal = (gen.statusFlags & (1 << 13)) > 0
    
    # Selection of products from Higgs decay chain (WWWW channel)
    is_lep = (abs(gen.pdgId) == 11) | (abs(gen.pdgId) == 13) | (abs(gen.pdgId) == 15)
    is_neu = (abs(gen.pdgId) == 12) | (abs(gen.pdgId) == 14) | (abs(gen.pdgId) == 16)
    is_quark = (abs(gen.pdgId) >= 1) & (abs(gen.pdgId) <= 5)
    
    # Identify gen-leptons from DiHiggs for calculation
    gen_leps_from_h = gen[is_from_h & is_terminal & is_lep]
    
    # Reconstruct HH from all terminal products of Higgses
    # (leptons + neutrinos + quarks from Higgs chain)
    gen_hh_products = gen[is_from_h & is_terminal & (is_lep | is_neu | is_quark)]
    gen_hh = ak.sum(gen_hh_products, axis=1)
    
    mHH_gen = ak.fill_none(gen_hh.mass, -999.0)
    
    # nFromDiHiggs: count total gen-leptons from Higgses
    nFromDiHiggs = ak.num(gen_leps_from_h, axis=1)

    # 5. Save Columns
    events = set_ak_column(events, "lep_isPrompt", lep_isPrompt)
    events = set_ak_column(events, "lep_isUnmatched", lep_isUnmatched)
    
    events = set_ak_column(events, "nFromDiHiggs", nFromDiHiggs)
    events = set_ak_column(events, "mHH_gen", mHH_gen)

    return events, SelectionResult(steps={}, objects={}, aux={})
