# This workflow has been generated by the pocket_coffea CLI 0.9.6.
import awkward as ak
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.objects import (
    jet_correction,
    lepton_selection,
    jet_selection,
    btagging,
)
import numba
import numpy as np

@numba.njit
def get_max_mjj(njets, mjjs):
    out = -1 * np.ones((len(njets), 2))
    mjj_best = np.ones(len(njets))
    for iev in range(len(njets)):
        max_mjj = -1.
        best_1 = -1
        best_2 = -2
        for i in range(njets[iev]-1):
            for j in range(i+1, njets[iev]):
                #print(i, j, mjjs[iev][i][j])
                if mjjs[iev][i][j] > max_mjj:
                    best_1 = i
                    best_2 = j
                    max_mjj =  mjjs[iev][i][j]
        out[iev][0] = best_1
        out[iev][1] = best_2
        mjj_best[iev] = max_mjj
    return out, mjj_best

def mjj(obj1, obj2):
    return (obj1 + obj2).mass


class SSWWProcessor(BaseProcessorABC):

    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def apply_object_preselection(self, variation):
        # Include the supercluster pseudorapidity variable
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        # Build masks for selection of muons, electrons, jets, fatjets
        self.events["MuonGood"] = lepton_selection(self.events, "Muon", self.params)
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        leptons =  ak.concatenate((self.events.MuonGood, self.events.ElectronGood), axis=1)
        self.events["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params,
            self._year,
            leptons_collection="LeptonGood"
        )
        self.events["BJetGood"] = btagging(
            self.events["JetGood"],
            self.params.btagging.working_point[self._year],
            wp=self.params.object_preselection.Jet.btag.wp
        )
        
        # Pad the LeptonGood collection to 2 objects
        self.events["LeptonGood"] = ak.with_name(
            ak.fill_none(
                ak.pad_none(
                    self.events["LeptonGood"], 2, axis=1),
                {"pt": 0., "eta": 0., "phi": 0., "mass": 0., "charge": 1., "pdgId": 0}
            ),
            name='PtEtaPhiMCandidate',
        )

      


    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nLeptonGood"] = (
            self.events["nMuonGood"] + self.events["nElectronGood"]
        )
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        self.events["nBJetGood"] = ak.num(self.events.BJetGood)
        # self.events["nfatjet"]   = ak.num(self.events.FatJetGood)

     

    def define_common_variables_after_presel(self, variation):
        # Defie Mjj vbs
        Mjj_all = self.events.Jet.metric_table(self.events.JetGood, metric=mjj)
        
        largest_mjj_idx, largest_mjj = get_max_mjj(
            ak.without_parameters(ak.num(self.events.JetGood, axis=1), behavior=None),
            ak.without_parameters(Mjj_all, behavior=None))
        
        self.events["mjj_vbs"] = largest_mjj
        
        loc_idk = ak.local_index(self.events.JetGood.eta)
        mask_vbs = (loc_idk == largest_mjj_idx[:,0]) | (loc_idk == largest_mjj_idx[:,1])
        self.events["VBSJets"] = self.events.JetGood[mask_vbs]
        self.events["nonVBSJets"] = self.events.JetGood[~mask_vbs]
        
        self.events["deltaeta_vbs"] = abs(self.events.VBSJets[:,0].eta - self.events.VBSJets[:,1].eta)

        # Zeppendfeld variable
        self.events["Zeppl1_vbs"] = abs(
            self.events.LeptonGood[:,0].eta
            - self.events.VBSJets[:,0].eta
            + self.events.VBSJets[:,1].eta)/ (2*self.events.deltaeta_vbs)
        self.events["Zeppl2_vbs"] = abs(
            self.events.LeptonGood[:,1].eta
            - self.events.VBSJets[:,0].eta
            + self.events.VBSJets[:,1].eta)/ (2*self.events.deltaeta_vbs)

        #mll
        self.events["mll"] = (self.events.LeptonGood[:,0] + self.events.LeptonGood[:,1]).mass
        

        # Defining quadrimoment of W and VBSpartons
        self.events["W1"] = self.events.LHEPart[:,2 ] + self.events.LHEPart[:,3]
        self.events["W2"] = self.events.LHEPart[:,4 ] + self.events.LHEPart[:,5]

        self.events["Neutrino1"] = self.events.LHEPart[:,3]
        self.events["Neutrino2"] = self.events.LHEPart[:,5]
        
        charge1 = np.sign(self.events.LHEPart[:,2].pdgId)
        charge2 = np.sign(self.events.LHEPart[:,4].pdgId)
        self.events["W1"] = ak.with_field(self.events.W1, charge1, "charge")
        self.events["W2"] = ak.with_field(self.events.W2, charge2, "charge")
        self.events["VBSPartons"] = self.events.LHEPart[:,6:8]
