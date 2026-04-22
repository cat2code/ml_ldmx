BRANCHES = {
    "ecal": {
        "simhits_pileup": {
            "scalars": [],
            "vectors": {
                "x": "EcalSimHits_pileup/EcalSimHits_pileup.x_",
                "y": "EcalSimHits_pileup/EcalSimHits_pileup.y_",
                "z": "EcalSimHits_pileup/EcalSimHits_pileup.z_",
                "energy": "EcalSimHits_pileup/EcalSimHits_pileup.edep_",
                "track_id_contribs": "EcalSimHits_pileup/EcalSimHits_pileup.track_id_contribs_",
                "edep_contribs": "EcalSimHits_pileup/EcalSimHits_pileup.edep_contribs_",
                "origin_id_contribs": "EcalSimHits_pileup/EcalSimHits_pileup.origin_contribs_",
                "n_contribs": "EcalSimHits_pileup/EcalSimHits_pileup.n_contribs_",
            },
        },
        "rechits_overlay": {
            "scalars": [],
            "vectors": {
                "x": "EcalRecHits_overlay/EcalRecHits_overlay.xpos_",
                "y": "EcalRecHits_overlay/EcalRecHits_overlay.ypos_",
                "z": "EcalRecHits_overlay/EcalRecHits_overlay.zpos_",
                "energy": "EcalRecHits_overlay/EcalRecHits_overlay.energy_",
                "id": "EcalRecHits_overlay/EcalRecHits_overlay.id_",
                "noise_flag": "EcalRecHits_overlay/EcalRecHits_overlay.is_noise_",
            },
        },
        "simhits_overlay": {
            "scalars": [],
            "vectors": {
                "id": "EcalSimHitsOverlay_overlay/EcalSimHitsOverlay_overlay.id_",
                "track_id_contribs": "EcalSimHitsOverlay_overlay/EcalSimHitsOverlay_overlay.track_id_contribs_",
                "edep_contribs": "EcalSimHitsOverlay_overlay/EcalSimHitsOverlay_overlay.edep_contribs_",
                "origin_id_contribs": "EcalSimHitsOverlay_overlay/EcalSimHitsOverlay_overlay.origin_contribs_",
                "n_contribs": "EcalSimHitsOverlay_overlay/EcalSimHitsOverlay_overlay.n_contribs_",
            },
        },
    }
}

def get_collection(detector: str, collection: str) -> dict:
    return BRANCHES[detector][collection]


def get_vector_branches(detector: str, collection: str) -> dict:
    return BRANCHES[detector][collection]["vectors"]


def get_scalar_branches(detector: str, collection: str) -> list:
    return BRANCHES[detector][collection]["scalars"]


def get_all_branch_names(detector: str, collection: str) -> list:
    cfg = BRANCHES[detector][collection]
    return cfg["scalars"] + list(cfg["vectors"].values())
