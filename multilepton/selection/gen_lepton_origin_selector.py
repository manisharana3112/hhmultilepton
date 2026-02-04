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
def gen_dihiggs_selector(self, events, **kwargs):
    """
    Vectorized Gen-Level Selector.
    Calculates Lepton Origin flags and mHH_gen without loops.
    """
    gen = events.GenPart
    
    # 1. Collect Analysis Leptons (E, Mu, Tau)
    # We use 'ak.concatenate' to merge them into a single loose collection of leptons
    # Note: This simply concatenates arrays, order is E -> Mu -> Tau
    electrons = events.Electron
    muons = events.Muon
    taus = events.Tau
    
    # Concatenate gen indices
    lep_gen_indices = ak.concatenate([electrons.genPartIdx, muons.genPartIdx, taus.genPartIdx], axis=1)
    
    # 2. Vectorized Matching Helper
    def get_origin_flags(gen_indices):
        # Result arrays initialized to False
        is_prompt = ak.zeros_like(gen_indices, dtype=bool)
        is_from_higgs = ak.zeros_like(gen_indices, dtype=bool)
        is_from_tau = ak.zeros_like(gen_indices, dtype=bool)
        
        # Mask for valid gen matches (indices >= 0)
        has_match = gen_indices >= 0
        
        # --- Prompt Check ---
        # Get flags for matched particles (masking invalid indices with 0/safe logic)
        # We use explicit masking to prevent index error
        valid_indices = ak.mask(gen_indices, has_match)
        matched_gen = gen[valid_indices]
        
        # Check status flags (Bit 0=Prompt, Bit 8=PromptTauDecayProduct)
        # Note: ak.fill_none handles the masked entries (unmatched leptons)
        flags = ak.fill_none(matched_gen.statusFlags, 0)
        is_prompt = (flags & 1) | (flags & (1 << 8)) > 0
        
        # --- Ancestry Tracing (Vectorized Loop) ---
        # We trace parents up to 15 generations
        
        current_indices = valid_indices # Start with direct match
        
        # Loop for ancestry depth
        for _ in range(15):
            # If current index is None or <0, stop tracing for that particle (handled by mask)
            # Fetch parent PDG
            # We need to handle the case where current_indices is None
            
            # Access Properties of current parent
            # Note: matched_gen already refers to 'current_indices' row
            # To iterate, we need to fetch the particle at 'current_indices'
            
            # Safe fetch
            current_parts = gen[current_indices]
            pdgs = ak.fill_none(abs(current_parts.pdgId), 0)
            
            # Check conditions
            is_from_higgs = is_from_higgs | (pdgs == 25)
            is_from_tau = is_from_tau | (pdgs == 15)
            
            # Update indices to mother
            parents = current_parts.genPartIdxMother
            # Update loop variables (propagate None if already reached end)
            current_indices = parents 

        return is_prompt, is_from_higgs, is_from_tau

    # Run the helper
    lep_isPrompt, lep_isFromHiggs, lep_isFromTau = get_origin_flags(lep_gen_indices)
    
    # Unmatched is simple
    lep_isUnmatched = lep_gen_indices < 0

    # 3. Di-Higgs Reconstruction (Vectorized)
    # Find Gen Higgs (PDG 25)
    higgses = gen[abs(gen.pdgId) == 25]
    
    # Check if we have at least 2 Higgs
    has_dihiggs = ak.num(higgses, axis=1) >= 2
    
    # Calculate mHH
    # Pad to 2 to allow calculation (fill None if < 2)
    h_pairs = ak.pad_none(higgses, 2)
    h1 = h_pairs[:, 0]
    h2 = h_pairs[:, 1]
    
    # We rely on vector behavior being attached to GenPart usually. 
    # If not, we can build it manually.
    # Assuming attach_coffea_behavior ran in default.py
    mHH_gen = (h1 + h2).mass
    mHH_gen = ak.fill_none(mHH_gen, -999.0) # Fill invalid
    
    # Count leptons from DiHiggs
    # Logic: Lepton is from Higgs AND Event has DiHiggs
    lep_isFromDiHiggs = lep_isFromHiggs & has_dihiggs
    nFromDiHiggs = ak.sum(lep_isFromDiHiggs, axis=1)

    # 4. Save Columns
    events = set_ak_column(events, "lep_isPrompt", lep_isPrompt)
    events = set_ak_column(events, "lep_isFromHiggs", lep_isFromHiggs)
    events = set_ak_column(events, "lep_isFromTau", lep_isFromTau)
    events = set_ak_column(events, "lep_isUnmatched", lep_isUnmatched)
    
    events = set_ak_column(events, "nFromDiHiggs", nFromDiHiggs)
    events = set_ak_column(events, "mHH_gen", mHH_gen)

    return events, SelectionResult(steps={}, objects={}, aux={})
