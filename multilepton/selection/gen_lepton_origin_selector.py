from columnflow.util import maybe_import

ak = maybe_import("awkward")
np = maybe_import("numpy")

from columnflow.selection import selector, Selector, SelectionResult
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

vector = maybe_import("vector")


@selector(
    uses={
        "GenPart.{pdgId,genPartIdxMother,statusFlags,pt,eta,phi,mass,px,py,pz,energy}",
        "Electron.genPartIdx",
        "Muon.genPartIdx",
        "Tau.genPartIdx",
    },
    produces={
        # Lepton origin classification
        "lep_isPrompt", "lep_isFromTau", "lep_isFromBOrC",
        "lep_isFromPhotonConv", "lep_isUnmatched",
        "lep_isFromHiggs",
        "lep_isFromHiggsZZ", "lep_isFromHiggsWW", "lep_isFromHiggsTauTau",
        "lep_isFromHiggsVV", "lep_isFromHiggsGG",
        "lep_isFromDiHiggs", "lep_isFromDiHiggsWWWW", "lep_isFromDiHiggsWWTauTau",
        "lep_isFromDiHiggsTauTauTauTau", "lep_isFromDiHiggsWWZZ",
        "lep_higgs_index",
        "nPrompt", "nFake", "nFromHiggs", "nFromDiHiggs",
        
        # Di-Higgs system variables for HH → WW WW channel
        "mHH_WWWW_gen", "pTHH_WWWW_gen", "dR_HH_WWWW_gen", "cosThetaStar_WWWW_gen",
        "H1_mass_WWWW_gen", "H2_mass_WWWW_gen",
        
        # Di-Higgs system variables for HH → WW ττ channel
        "mHH_WWTauTau_gen", "pTHH_WWTauTau_gen", "dR_HH_WWTauTau_gen", "cosThetaStar_WWTauTau_gen",
        "H1_mass_WWTauTau_gen", "H2_mass_WWTauTau_gen",
        
        # Di-Higgs system variables for HH → ττ ττ channel
        "mHH_TauTauTauTau_gen", "pTHH_TauTauTauTau_gen", "dR_HH_TauTauTauTau_gen", "cosThetaStar_TauTauTauTau_gen",
        "H1_mass_TauTauTauTau_gen", "H2_mass_TauTauTauTau_gen",
        
        # Di-Higgs system variables for HH → WW ZZ channel
        "mHH_WWZZ_gen", "pTHH_WWZZ_gen", "dR_HH_WWZZ_gen", "cosThetaStar_WWZZ_gen",
        "H1_mass_WWZZ_gen", "H2_mass_WWZZ_gen",
        
        # Topology validation
        "has_valid_gen_topology",
        "n_gen_leptons_from_W",
        "n_gen_neutrinos_from_W",
    },
    exposed=False,
)
def gen_dihiggs_selector(self: Selector, events: ak.Array, **kwargs):
    """
    MERGED GEN-LEVEL SELECTOR: LEPTON ORIGIN + DIHIGGS RECONSTRUCTION
    -----------------------------------------------------------------------
    This selector combines:
    1. Lepton origin classification (from gen_lepton_origin_selector)
    2. Di-Higgs system variable calculation (from gen_higgs_builder)
    
    Key Features:
    - Annotates leptons with their origin (Prompt, FromHiggs, FromDiHiggs, etc.)
    - Calculates di-Higgs system variables (mHH, pTHH, dR_HH, etc.)
    - Does NOT calculate individual lepton kinematics (only system-level variables)
    """

    gen = events.GenPart

    # ========================================================================
    # HELPER FUNCTIONS FOR LEPTON ORIGIN CLASSIFICATION
    # ========================================================================
    
    def get_parent_pdg(idx, gen_ev):
        if idx is None or idx < 0:
            return 0
        try:
            mom_idx = gen_ev.genPartIdxMother[idx]
            if mom_idx is None or mom_idx < 0:
                return 0
            return abs(gen_ev.pdgId[mom_idx])
        except Exception:
            return 0

    def get_ultimate_ancestor_pdg(idx, gen_ev):
        if idx is None or idx < 0:
            return 0
        cur = idx
        for _ in range(30):
            try:
                pdg = abs(gen_ev.pdgId[cur])
                mom = gen_ev.genPartIdxMother[cur]
            except Exception:
                return 0
            if mom is None or mom < 0 or mom == cur:
                return pdg
            cur = mom
        return pdg

    def get_higgs_decay_channel(idx, gen_ev):
        """
        Traces lepton back to Higgs and identifies decay channel.
        Returns: (is_from_higgs, decay_chain, higgs_idx)
        """
        if idx is None or idx < 0:
            return False, [], -1

        cur = idx
        decay_chain = []

        for _ in range(30):
            try:
                pdg = abs(gen_ev.pdgId[cur])
                mom_idx = gen_ev.genPartIdxMother[cur]
            except Exception:
                return False, [], -1

            decay_chain.append(pdg)

            if mom_idx is None or mom_idx < 0 or mom_idx == cur:
                return False, decay_chain, -1

            try:
                mom_pdg = abs(gen_ev.pdgId[mom_idx])
            except Exception:
                return False, decay_chain, -1

            if mom_pdg == 25:
                return True, decay_chain, mom_idx

            cur = mom_idx

        return False, decay_chain, -1

    def classify_higgs_channel(decay_chain):
        """
        Input: decay_chain = list of PDGs from lepton to Higgs
        Output: tuple of (is_ZZ, is_WW, is_TauTau, is_VV, is_GG)
        """
        is_zz = False
        is_ww = False
        is_tautau = False
        is_vv = False
        is_gg = False

        has_z = 23 in decay_chain
        has_w = 24 in decay_chain
        has_tau = 15 in decay_chain
        has_gluon = 21 in decay_chain

        if has_z and not has_w and not has_tau:
            is_zz = True
            is_vv = True
        elif has_w and not has_z and not has_tau:
            is_ww = True
            is_vv = True
        elif has_tau and not has_z and not has_w:
            is_tautau = True
        elif has_z and has_w:
            is_vv = True
        elif has_gluon:
            is_gg = True

        return is_zz, is_ww, is_tautau, is_vv, is_gg

    def find_higgs_bosons(gen_ev):
        """
        Find all Higgs bosons (PDG=25) in the event.
        Returns: list of Higgs indices
        """
        higgs_indices = []
        for i in range(len(gen_ev.pdgId)):
            if abs(gen_ev.pdgId[i]) == 25:
                higgs_indices.append(i)
        return higgs_indices

    def classify_dihiggs_channel(h1_idx, h2_idx, gen_ev):
        """
        Classify the DiHiggs decay channel based on the daughters of both Higgs.
        Returns: (is_WWWW, is_WWTauTau, is_TauTauTauTau, is_WWZZ)
        """
        def get_higgs_daughters(h_idx):
            """Get the daughter particles of a Higgs boson"""
            daughters = []
            for i in range(len(gen_ev.pdgId)):
                try:
                    mom_idx = gen_ev.genPartIdxMother[i]
                    if mom_idx == h_idx:
                        daughters.append(abs(gen_ev.pdgId[i]))
                except Exception:
                    pass
            return daughters

        h1_daughters = get_higgs_daughters(h1_idx)
        h2_daughters = get_higgs_daughters(h2_idx)

        # Count W, Z and tau daughters
        h1_has_w = 24 in h1_daughters
        h1_has_tau = 15 in h1_daughters
        h1_has_z = 23 in h1_daughters
        h2_has_w = 24 in h2_daughters
        h2_has_tau = 15 in h2_daughters
        h2_has_z = 23 in h2_daughters

        # Classify combinations
        is_wwww = h1_has_w and h2_has_w
        is_wwtautau = (h1_has_w and h2_has_tau) or (h1_has_tau and h2_has_w)
        is_tautautautau = h1_has_tau and h2_has_tau
        is_wwzz = (h1_has_w and h2_has_z) or (h1_has_z and h2_has_w)

        return is_wwww, is_wwtautau, is_tautautautau, is_wwzz

    # ========================================================================
    # HELPER FUNCTIONS FOR DIHIGGS RECONSTRUCTION
    # ========================================================================
    
    def build_4vector(particle):
        """Build a 4-vector from a GenPart particle"""
        try:
            return vector.obj(
                px=float(particle.px),
                py=float(particle.py),
                pz=float(particle.pz),
                E=float(particle.energy)
            )
        except (AttributeError, TypeError):
            return vector.obj(
                pt=float(particle.pt),
                eta=float(particle.eta),
                phi=float(particle.phi),
                mass=float(particle.mass)
            )

    def is_prompt_lepton(particle):
        """Check if particle is a prompt lepton"""
        try:
            flags = int(particle.statusFlags)
            is_prompt = (flags & 1) or (flags & (1 << 8))
            return bool(is_prompt)
        except (AttributeError, TypeError):
            return True

    def find_mother_particle(particle, genparts):
        """Find the mother particle of a given GenPart particle"""
        try:
            mother_idx = int(particle.genPartIdxMother)
            if mother_idx >= 0 and mother_idx < len(genparts):
                return genparts[mother_idx]
            else:
                return None
        except (AttributeError, TypeError, IndexError):
            return None

    def pair_leptons_and_neutrinos(leptons, neutrinos, genparts):
        """
        Pair leptons with their corresponding neutrinos from W decays.
        Returns: (lepton1, neutrino1, lepton2, neutrino2) or (None, None, None, None)
        """
        if len(leptons) != 2 or len(neutrinos) != 2:
            return None, None, None, None

        # Try mother-based matching first
        lep1_mother = find_mother_particle(leptons[0], genparts)
        lep2_mother = find_mother_particle(leptons[1], genparts)
        nu1_mother = find_mother_particle(neutrinos[0], genparts)
        nu2_mother = find_mother_particle(neutrinos[1], genparts)

        # Check if we can match by mother PDG ID (should be W boson = ±24)
        if lep1_mother is not None and nu1_mother is not None:
            try:
                if abs(lep1_mother.pdgId) == 24 and abs(nu1_mother.pdgId) == 24:
                    return leptons[0], neutrinos[0], leptons[1], neutrinos[1]
            except (AttributeError, TypeError):
                pass

        # Fallback: flavor matching (e⁺ with νₑ, μ⁺ with νμ)
        for nu in neutrinos:
            nu_pdg = abs(int(nu.pdgId))
            for lep in leptons:
                lep_pdg = abs(int(lep.pdgId))
                if (lep_pdg == 11 and nu_pdg == 12) or (lep_pdg == 13 and nu_pdg == 14):
                    other_lep = leptons[1] if lep is leptons[0] else leptons[0]
                    other_nu = neutrinos[1] if nu is neutrinos[0] else neutrinos[0]
                    return lep, nu, other_lep, other_nu

        # If all else fails, just pair in order
        return leptons[0], neutrinos[0], leptons[1], neutrinos[1]

    # ========================================================================
    # INITIALIZE OUTPUT ARRAYS
    # ========================================================================
    
    n_events = len(events)
    
    # Lepton origin classification arrays
    lep_isPrompt_all = []
    lep_isFromTau_all = []
    lep_isFromBOrC_all = []
    lep_isFromPhotonConv_all = []
    lep_isUnmatched_all = []
    lep_isFromHiggs_all = []
    lep_isFromHiggsZZ_all = []
    lep_isFromHiggsWW_all = []
    lep_isFromHiggsTauTau_all = []
    lep_isFromHiggsVV_all = []
    lep_isFromHiggsGG_all = []
    lep_isFromDiHiggs_all = []
    lep_isFromDiHiggsWWWW_all = []
    lep_isFromDiHiggsWWTauTau_all = []
    lep_isFromDiHiggsTauTauTauTau_all = []
    lep_isFromDiHiggsWWZZ_all = []
    lep_higgs_index_all = []
    nPrompt_list = []
    nFake_list = []
    nFromHiggs_list = []
    nFromDiHiggs_list = []
    
    # Di-Higgs system variables for HH → WW WW
    mHH_WWWW_gen = np.full(n_events, np.nan, dtype=np.float32)
    pTHH_WWWW_gen = np.full(n_events, np.nan, dtype=np.float32)
    dR_HH_WWWW_gen = np.full(n_events, np.nan, dtype=np.float32)
    cosThetaStar_WWWW_gen = np.full(n_events, np.nan, dtype=np.float32)
    H1_mass_WWWW_gen = np.full(n_events, np.nan, dtype=np.float32)
    H2_mass_WWWW_gen = np.full(n_events, np.nan, dtype=np.float32)
    
    # Di-Higgs system variables for HH → WW ττ
    mHH_WWTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    pTHH_WWTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    dR_HH_WWTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    cosThetaStar_WWTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    H1_mass_WWTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    H2_mass_WWTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    
    # Di-Higgs system variables for HH → ττ ττ
    mHH_TauTauTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    pTHH_TauTauTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    dR_HH_TauTauTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    cosThetaStar_TauTauTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    H1_mass_TauTauTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    H2_mass_TauTauTauTau_gen = np.full(n_events, np.nan, dtype=np.float32)
    
    # Di-Higgs system variables for HH → WW ZZ
    mHH_WWZZ_gen = np.full(n_events, np.nan, dtype=np.float32)
    pTHH_WWZZ_gen = np.full(n_events, np.nan, dtype=np.float32)
    dR_HH_WWZZ_gen = np.full(n_events, np.nan, dtype=np.float32)
    cosThetaStar_WWZZ_gen = np.full(n_events, np.nan, dtype=np.float32)
    H1_mass_WWZZ_gen = np.full(n_events, np.nan, dtype=np.float32)
    H2_mass_WWZZ_gen = np.full(n_events, np.nan, dtype=np.float32)
    
    # Topology validation
    has_valid_gen_topology = np.zeros(n_events, dtype=bool)
    n_gen_leptons_from_W = np.zeros(n_events, dtype=np.int32)
    n_gen_neutrinos_from_W = np.zeros(n_events, dtype=np.int32)

    # ========================================================================
    # EVENT LOOP: Process each event
    # ========================================================================
    
    for iev in range(n_events):
        gen_ev = gen[iev]

        # --------------------------------------------------------------------
        # PART 1: LEPTON ORIGIN CLASSIFICATION
        # --------------------------------------------------------------------
        
        # Find Higgs bosons in this event
        higgs_indices = find_higgs_bosons(gen_ev)
        
        # Check for DiHiggs event
        is_dihiggs_event = len(higgs_indices) >= 2
        is_hh_wwww = False
        is_hh_wwtautau = False
        is_hh_tautautautau = False
        is_hh_wwzz = False

        if is_dihiggs_event:
            h1_idx = higgs_indices[0]
            h2_idx = higgs_indices[1]
            
            # Classify DiHiggs channel
            is_hh_wwww, is_hh_wwtautau, is_hh_tautautautau, is_hh_wwzz = classify_dihiggs_channel(
                h1_idx, h2_idx, gen_ev
            )

        # Process leptons for origin classification
        prompt_flags = []
        tau_flags = []
        bc_flags = []
        phconv_flags = []
        unmatched_flags = []
        higgs_flags = []
        higgs_zz_flags = []
        higgs_ww_flags = []
        higgs_tautau_flags = []
        higgs_vv_flags = []
        higgs_gg_flags = []
        dihiggs_flags = []
        dihiggs_wwww_flags = []
        dihiggs_wwtautau_flags = []
        dihiggs_tautautautau_flags = []
        dihiggs_wwzz_flags = []
        higgs_idx_flags = []

        all_lepton_gidx = list(ak.to_list(events.Electron.genPartIdx[iev])) + \
                          list(ak.to_list(events.Muon.genPartIdx[iev])) + \
                          list(ak.to_list(events.Tau.genPartIdx[iev]))

        for gidx in all_lepton_gidx:
            if gidx is None or gidx < 0:
                prompt_flags.append(False)
                tau_flags.append(False)
                bc_flags.append(False)
                phconv_flags.append(False)
                unmatched_flags.append(True)
                higgs_flags.append(False)
                higgs_zz_flags.append(False)
                higgs_ww_flags.append(False)
                higgs_tautau_flags.append(False)
                higgs_vv_flags.append(False)
                higgs_gg_flags.append(False)
                dihiggs_flags.append(False)
                dihiggs_wwww_flags.append(False)
                dihiggs_wwtautau_flags.append(False)
                dihiggs_tautautautau_flags.append(False)
                dihiggs_wwzz_flags.append(False)
                higgs_idx_flags.append(-1)
                continue

            # Get ultimate ancestor
            anc = get_ultimate_ancestor_pdg(int(gidx), gen_ev)

            # Basic classification
            is_prompt = anc in (24, 23, 15, 25)
            is_tau = anc == 15
            is_bc = anc in (4, 5)
            is_conv = anc == 22
            is_higgs = anc == 25

            prompt_flags.append(is_prompt)
            tau_flags.append(is_tau)
            bc_flags.append(is_bc)
            phconv_flags.append(is_conv)
            unmatched_flags.append(False)
            higgs_flags.append(is_higgs)

            # Single Higgs decay channel classification
            is_from_h, decay_chain, h_idx = get_higgs_decay_channel(int(gidx), gen_ev)
            
            if is_from_h:
                is_zz, is_ww, is_tautau, is_vv, is_gg = classify_higgs_channel(decay_chain)
                higgs_idx_flags.append(h_idx)
            else:
                is_zz = is_ww = is_tautau = is_vv = is_gg = False
                higgs_idx_flags.append(-1)

            higgs_zz_flags.append(is_zz)
            higgs_ww_flags.append(is_ww)
            higgs_tautau_flags.append(is_tautau)
            higgs_vv_flags.append(is_vv)
            higgs_gg_flags.append(is_gg)

            # DiHiggs classification
            is_from_dihiggs = is_dihiggs_event and is_from_h
            dihiggs_flags.append(is_from_dihiggs)
            
            if is_from_dihiggs:
                dihiggs_wwww_flags.append(is_hh_wwww)
                dihiggs_wwtautau_flags.append(is_hh_wwtautau)
                dihiggs_tautautautau_flags.append(is_hh_tautautautau)
                dihiggs_wwzz_flags.append(is_hh_wwzz)
            else:
                dihiggs_wwww_flags.append(False)
                dihiggs_wwtautau_flags.append(False)
                dihiggs_tautautautau_flags.append(False)
                dihiggs_wwzz_flags.append(False)

        # Event-level counts
        nPrompt = sum(prompt_flags)
        nFake = sum(
            (not prompt_flags[i]) and (not unmatched_flags[i])
            for i in range(len(prompt_flags))
        )
        nFromHiggs = sum(higgs_flags)
        nFromDiHiggs = sum(dihiggs_flags)

        lep_isPrompt_all.append(prompt_flags)
        lep_isFromTau_all.append(tau_flags)
        lep_isFromBOrC_all.append(bc_flags)
        lep_isFromPhotonConv_all.append(phconv_flags)
        lep_isUnmatched_all.append(unmatched_flags)
        lep_isFromHiggs_all.append(higgs_flags)
        lep_isFromHiggsZZ_all.append(higgs_zz_flags)
        lep_isFromHiggsWW_all.append(higgs_ww_flags)
        lep_isFromHiggsTauTau_all.append(higgs_tautau_flags)
        lep_isFromHiggsVV_all.append(higgs_vv_flags)
        lep_isFromHiggsGG_all.append(higgs_gg_flags)
        lep_isFromDiHiggs_all.append(dihiggs_flags)
        lep_isFromDiHiggsWWWW_all.append(dihiggs_wwww_flags)
        lep_isFromDiHiggsWWTauTau_all.append(dihiggs_wwtautau_flags)
        lep_isFromDiHiggsTauTauTauTau_all.append(dihiggs_tautautautau_flags)
        lep_isFromDiHiggsWWZZ_all.append(dihiggs_wwzz_flags)
        lep_higgs_index_all.append(higgs_idx_flags)
        nPrompt_list.append(nPrompt)
        nFake_list.append(nFake)
        nFromHiggs_list.append(nFromHiggs)
        nFromDiHiggs_list.append(nFromDiHiggs)

        # --------------------------------------------------------------------
        # PART 2: DIHIGGS RECONSTRUCTION (for ALL decay channels)
        # --------------------------------------------------------------------
        
        if not is_dihiggs_event:
            continue
            
        try:
            genparts = gen_ev
        except (AttributeError, IndexError):
            continue
        
        if genparts is None or len(genparts) == 0:
            continue
        
        # Get the two Higgs bosons directly from GenPart
        h1_idx = higgs_indices[0]
        h2_idx = higgs_indices[1]
        
        try:
            h1_particle = genparts[h1_idx]
            h2_particle = genparts[h2_idx]
        except (IndexError, AttributeError):
            continue
        
        # Build 4-vectors for both Higgs bosons
        try:
            H1_vec = build_4vector(h1_particle)
            H2_vec = build_4vector(h2_particle)
        except Exception:
            continue
        
        # Reconstruct di-Higgs system
        HH_vec = H1_vec + H2_vec
        
        # Calculate di-Higgs kinematic variables
        mHH = float(HH_vec.mass)
        pTHH = float(HH_vec.pt)
        H1_mass = float(H1_vec.mass)
        H2_mass = float(H2_vec.mass)
        
        # ΔR separation between two Higgs
        try:
            dR_HH = float(H1_vec.deltaR(H2_vec))
        except Exception:
            dR_HH = np.nan
        
        # Calculate cos(θ*) - angle of H₁ in HH rest frame
        try:
            H1_rest = H1_vec.boost(-HH_vec.beta3)
            cosThetaStar = float(np.cos(H1_rest.theta))
        except Exception:
            cosThetaStar = np.nan
        
        # --------------------------------------------------------------------
        # Store variables in channel-specific arrays
        # --------------------------------------------------------------------
        
        if is_hh_wwww:
            # HH → WW WW channel
            mHH_WWWW_gen[iev] = mHH
            pTHH_WWWW_gen[iev] = pTHH
            dR_HH_WWWW_gen[iev] = dR_HH
            cosThetaStar_WWWW_gen[iev] = cosThetaStar
            H1_mass_WWWW_gen[iev] = H1_mass
            H2_mass_WWWW_gen[iev] = H2_mass
            
        elif is_hh_wwtautau:
            # HH → WW ττ channel
            mHH_WWTauTau_gen[iev] = mHH
            pTHH_WWTauTau_gen[iev] = pTHH
            dR_HH_WWTauTau_gen[iev] = dR_HH
            cosThetaStar_WWTauTau_gen[iev] = cosThetaStar
            H1_mass_WWTauTau_gen[iev] = H1_mass
            H2_mass_WWTauTau_gen[iev] = H2_mass
            
        elif is_hh_tautautautau:
            # HH → ττ ττ channel
            mHH_TauTauTauTau_gen[iev] = mHH
            pTHH_TauTauTauTau_gen[iev] = pTHH
            dR_HH_TauTauTauTau_gen[iev] = dR_HH
            cosThetaStar_TauTauTauTau_gen[iev] = cosThetaStar
            H1_mass_TauTauTauTau_gen[iev] = H1_mass
            H2_mass_TauTauTauTau_gen[iev] = H2_mass
            
        elif is_hh_wwzz:
            # HH → WW ZZ channel
            mHH_WWZZ_gen[iev] = mHH
            pTHH_WWZZ_gen[iev] = pTHH
            dR_HH_WWZZ_gen[iev] = dR_HH
            cosThetaStar_WWZZ_gen[iev] = cosThetaStar
            H1_mass_WWZZ_gen[iev] = H1_mass
            H2_mass_WWZZ_gen[iev] = H2_mass
        
        # --------------------------------------------------------------------
        # OPTIONAL: Count leptons and neutrinos from W decays (for validation)
        # --------------------------------------------------------------------
        
        # Find all charged leptons (e, μ) from W decays
        leptons = []
        for part in genparts:
            try:
                pdg = int(part.pdgId)
                abs_pdg = abs(pdg)
                
                if abs_pdg == 11 or abs_pdg == 13:
                    if is_prompt_lepton(part):
                        mother = find_mother_particle(part, genparts)
                        if mother is not None and abs(int(mother.pdgId)) == 24:
                            leptons.append(part)
            except (AttributeError, TypeError):
                continue
        
        n_gen_leptons_from_W[iev] = len(leptons)
        
        # Find all neutrinos (νₑ, νμ) from W decays
        neutrinos = []
        for part in genparts:
            try:
                pdg = int(part.pdgId)
                abs_pdg = abs(pdg)
                
                if abs_pdg == 12 or abs_pdg == 14:
                    if is_prompt_lepton(part):
                        mother = find_mother_particle(part, genparts)
                        if mother is not None and abs(int(mother.pdgId)) == 24:
                            neutrinos.append(part)
            except (AttributeError, TypeError):
                continue
        
        n_gen_neutrinos_from_W[iev] = len(neutrinos)
        
        # Mark topology as valid if we have 2 leptons + 2 neutrinos
        if len(leptons) == 2 and len(neutrinos) == 2:
            has_valid_gen_topology[iev] = True
        else:
            has_valid_gen_topology[iev] = False

    # ========================================================================
    # WRITE ALL OUTPUTS TO EVENTS
    # ========================================================================
    
    # Lepton origin classification
    events = set_ak_column(events, "lep_isPrompt", ak.Array(lep_isPrompt_all))
    events = set_ak_column(events, "lep_isFromTau", ak.Array(lep_isFromTau_all))
    events = set_ak_column(events, "lep_isFromBOrC", ak.Array(lep_isFromBOrC_all))
    events = set_ak_column(events, "lep_isFromPhotonConv", ak.Array(lep_isFromPhotonConv_all))
    events = set_ak_column(events, "lep_isUnmatched", ak.Array(lep_isUnmatched_all))
    events = set_ak_column(events, "lep_isFromHiggs", ak.Array(lep_isFromHiggs_all))
    events = set_ak_column(events, "lep_isFromHiggsZZ", ak.Array(lep_isFromHiggsZZ_all))
    events = set_ak_column(events, "lep_isFromHiggsWW", ak.Array(lep_isFromHiggsWW_all))
    events = set_ak_column(events, "lep_isFromHiggsTauTau", ak.Array(lep_isFromHiggsTauTau_all))
    events = set_ak_column(events, "lep_isFromHiggsVV", ak.Array(lep_isFromHiggsVV_all))
    events = set_ak_column(events, "lep_isFromHiggsGG", ak.Array(lep_isFromHiggsGG_all))
    events = set_ak_column(events, "lep_isFromDiHiggs", ak.Array(lep_isFromDiHiggs_all))
    events = set_ak_column(events, "lep_isFromDiHiggsWWWW", ak.Array(lep_isFromDiHiggsWWWW_all))
    events = set_ak_column(events, "lep_isFromDiHiggsWWTauTau", ak.Array(lep_isFromDiHiggsWWTauTau_all))
    events = set_ak_column(events, "lep_isFromDiHiggsTauTauTauTau", ak.Array(lep_isFromDiHiggsTauTauTauTau_all))
    events = set_ak_column(events, "lep_isFromDiHiggsWWZZ", ak.Array(lep_isFromDiHiggsWWZZ_all))
    events = set_ak_column(events, "lep_higgs_index", ak.Array(lep_higgs_index_all))

    # Event-level counts
    events = set_ak_column(events, "nPrompt", ak.Array(nPrompt_list))
    events = set_ak_column(events, "nFake", ak.Array(nFake_list))
    events = set_ak_column(events, "nFromHiggs", ak.Array(nFromHiggs_list))
    events = set_ak_column(events, "nFromDiHiggs", ak.Array(nFromDiHiggs_list))
    
    # Di-Higgs system variables for HH → WW WW
    events = set_ak_column(events, "mHH_WWWW_gen", ak.Array(mHH_WWWW_gen))
    events = set_ak_column(events, "pTHH_WWWW_gen", ak.Array(pTHH_WWWW_gen))
    events = set_ak_column(events, "dR_HH_WWWW_gen", ak.Array(dR_HH_WWWW_gen))
    events = set_ak_column(events, "cosThetaStar_WWWW_gen", ak.Array(cosThetaStar_WWWW_gen))
    events = set_ak_column(events, "H1_mass_WWWW_gen", ak.Array(H1_mass_WWWW_gen))
    events = set_ak_column(events, "H2_mass_WWWW_gen", ak.Array(H2_mass_WWWW_gen))
    
    # Di-Higgs system variables for HH → WW ττ
    events = set_ak_column(events, "mHH_WWTauTau_gen", ak.Array(mHH_WWTauTau_gen))
    events = set_ak_column(events, "pTHH_WWTauTau_gen", ak.Array(pTHH_WWTauTau_gen))
    events = set_ak_column(events, "dR_HH_WWTauTau_gen", ak.Array(dR_HH_WWTauTau_gen))
    events = set_ak_column(events, "cosThetaStar_WWTauTau_gen", ak.Array(cosThetaStar_WWTauTau_gen))
    events = set_ak_column(events, "H1_mass_WWTauTau_gen", ak.Array(H1_mass_WWTauTau_gen))
    events = set_ak_column(events, "H2_mass_WWTauTau_gen", ak.Array(H2_mass_WWTauTau_gen))
    
    # Di-Higgs system variables for HH → ττ ττ
    events = set_ak_column(events, "mHH_TauTauTauTau_gen", ak.Array(mHH_TauTauTauTau_gen))
    events = set_ak_column(events, "pTHH_TauTauTauTau_gen", ak.Array(pTHH_TauTauTauTau_gen))
    events = set_ak_column(events, "dR_HH_TauTauTauTau_gen", ak.Array(dR_HH_TauTauTauTau_gen))
    events = set_ak_column(events, "cosThetaStar_TauTauTauTau_gen", ak.Array(cosThetaStar_TauTauTauTau_gen))
    events = set_ak_column(events, "H1_mass_TauTauTauTau_gen", ak.Array(H1_mass_TauTauTauTau_gen))
    events = set_ak_column(events, "H2_mass_TauTauTauTau_gen", ak.Array(H2_mass_TauTauTauTau_gen))
    
    # Di-Higgs system variables for HH → WW ZZ
    events = set_ak_column(events, "mHH_WWZZ_gen", ak.Array(mHH_WWZZ_gen))
    events = set_ak_column(events, "pTHH_WWZZ_gen", ak.Array(pTHH_WWZZ_gen))
    events = set_ak_column(events, "dR_HH_WWZZ_gen", ak.Array(dR_HH_WWZZ_gen))
    events = set_ak_column(events, "cosThetaStar_WWZZ_gen", ak.Array(cosThetaStar_WWZZ_gen))
    events = set_ak_column(events, "H1_mass_WWZZ_gen", ak.Array(H1_mass_WWZZ_gen))
    events = set_ak_column(events, "H2_mass_WWZZ_gen", ak.Array(H2_mass_WWZZ_gen))
    
    # Topology validation
    events = set_ak_column(events, "has_valid_gen_topology", ak.Array(has_valid_gen_topology))
    events = set_ak_column(events, "n_gen_leptons_from_W", ak.Array(n_gen_leptons_from_W))
    events = set_ak_column(events, "n_gen_neutrinos_from_W", ak.Array(n_gen_neutrinos_from_W))

    return events, SelectionResult(steps={}, objects={}, aux={})
