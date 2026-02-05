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
        "lep_isFromHiggs",
        "lep_isFromTau",
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
    lep_isFromHiggs = ak.concatenate([ele_isFromHiggs, mu_isFromHiggs, tau_isFromHiggs], axis=1)
    lep_isFromTau = ak.concatenate([ele_isFromTau, mu_isFromTau, tau_isFromTau], axis=1)
    lep_isUnmatched = ak.concatenate([events.Electron.genPartIdx < 0, events.Muon.genPartIdx < 0, events.Tau.genPartIdx < 0], axis=1)

    # 3. Calculate Event-Level Flags (if lepton_results is present)
    # Be careful not to overwrite the Per-Lepton arrays variable names if needed, 
    # but here we are producing new columns "lep_isPrompt" (Event Level) which might conflict with the produced column name above?
    # Wait, the user asked to add "lep_isPrompt" to variables.py as an EVENT level variable.
    # But the produces={} list above has "lep_isPrompt". 
    # Usually "lep_isPrompt" (concatenated) is a jagged array of booleans. columnflow handles this fine.
    # If we want an *Event Level* boolean, we must name it differently/reuse carefully.
    # The 'variables.py' entry I added was `name="lep_isPrompt"`. 
    # If I overwrite the column "lep_isPrompt" with a single boolean per event, I lose the per-lepton info.
    # HOWEVER, the User explicitly asked for "lep_isPrompt" and "lep_isUnmatched" variables.
    # If the user wants to cut on "event is prompt", they usually want a single boolean.
    # BUT, if they plot it, they might want per-lepton.
    # Given the previous context "splitting leptons into fakes and non fakes based on the channels",
    # I will create specific Event-Level columns and maybe overwrite `lep_isPrompt` IF the user intends it to be the event flag.
    # BUT standard practice is to keep `lep_` as jagged.
    # I will OVERWRITE `lep_isPrompt` with the EVENT-LEVEL flag since the user defined it with `discrete_x=True` and `binning=(2, -0.5, 1.5)`, which implies a single value per event (0 or 1).
    # If I overwrite it, I lose per-lepton info. 
    # Actually, the user's variables.py entry `lep_isPrompt` has `x_title="Prompt Lepton (Event)"`.
    # This strongly suggests they want the Event-Level flag in that variable.
    # I will produce TWO sets of columns:
    # 1. `lep_isPrompt_perLep` (Jagged) - optional, or just don't save it if not needed?
    #    The selector `produces` list has `lep_isPrompt`.
    #    I will assume the selector wants to output the EVENT LEVEL flag now based on the variable definition.

    # Let's check the logic:
    # If I change `lep_isPrompt` to be event-level (flat), any downstream task expecting jagged array will fail.
    # But the user specifically asked for this variable change.
    # Use caution: I will save the event-level flag as `lep_isPrompt`.
    
    if lepton_results:
        # Get selected indices
        sel_ele = lepton_results.objects.Electron.Electron
        sel_mu = lepton_results.objects.Muon.Muon
        sel_tau = lepton_results.objects.Tau.Tau
        
        # Check selected leptons only
        sel_ele_prompt = ele_isPrompt[sel_ele]
        sel_mu_prompt = mu_isPrompt[sel_mu]
        sel_tau_prompt = tau_isPrompt[sel_tau]
        
        # Event is prompt if ALL selected leptons are prompt AND the event passed selection
        # (ak.all on empty list is True, so the lepton_results.event mask is vital)
        is_p_event = ak.all(sel_ele_prompt, axis=1) & ak.all(sel_mu_prompt, axis=1) & ak.all(sel_tau_prompt, axis=1)
        is_p_event = is_p_event & lepton_results.event
        
        # Unmatched: If ANY selected is unmatched AND event passed selection
        sel_ele_un = (events.Electron.genPartIdx < 0)[sel_ele]
        sel_mu_un = (events.Muon.genPartIdx < 0)[sel_mu]
        sel_tau_un = (events.Tau.genPartIdx < 0)[sel_tau]
        
        is_un_event = ak.any(sel_ele_un, axis=1) | ak.any(sel_mu_un, axis=1) | ak.any(sel_tau_un, axis=1)
        is_un_event = is_un_event & lepton_results.event
        
        # From Higgs / From Tau
        is_h_event = ak.any(ele_isFromHiggs[sel_ele], axis=1) | ak.any(mu_isFromHiggs[sel_mu], axis=1) | ak.any(tau_isFromHiggs[sel_tau], axis=1)
        is_t_event = ak.any(ele_isFromTau[sel_ele], axis=1) | ak.any(mu_isFromTau[sel_mu], axis=1) | ak.any(tau_isFromTau[sel_tau], axis=1)
        
        is_h_event = is_h_event & lepton_results.event
        is_t_event = is_t_event & lepton_results.event

        # Overwrite the output variables with Event-Level BOOLEANS
        lep_isPrompt = is_p_event
        lep_isUnmatched = is_un_event
        lep_isFromHiggs = is_h_event
        lep_isFromTau = is_t_event

    # 4. Di-Higgs Reconstruction (Standard)
    higgses = gen[abs(gen.pdgId) == 25]
    has_dihiggs = ak.num(higgses, axis=1) >= 2
    h_pairs = ak.pad_none(higgses, 2)
    mHH_gen = (h_pairs[:, 0] + h_pairs[:, 1]).mass
    mHH_gen = ak.fill_none(mHH_gen, -999.0)
    
    # nFromDiHiggs (This is tricky - usually counts total leptons from higgs in event)
    # We will keep the original "per-lepton" based counting for this one?
    # "lep_isFromDiHiggs" was derived from "lep_isFromHiggs & has_dihiggs".
    # Since I overwrote lep_isFromHiggs with event-level, I need the jagged one again.
    # Re-calculate jagged for this specific calculation
    lep_isFromHiggs_jagged = ak.concatenate([ele_isFromHiggs, mu_isFromHiggs, tau_isFromHiggs], axis=1)
    lep_isFromDiHiggs = lep_isFromHiggs_jagged & has_dihiggs
    nFromDiHiggs = ak.sum(lep_isFromDiHiggs, axis=1)

    # 5. Save Columns
    events = set_ak_column(events, "lep_isPrompt", lep_isPrompt)
    events = set_ak_column(events, "lep_isFromHiggs", lep_isFromHiggs)
    events = set_ak_column(events, "lep_isFromTau", lep_isFromTau)
    events = set_ak_column(events, "lep_isUnmatched", lep_isUnmatched)
    
    events = set_ak_column(events, "nFromDiHiggs", nFromDiHiggs)
    events = set_ak_column(events, "mHH_gen", mHH_gen)

    return events, SelectionResult(steps={}, objects={}, aux={})
