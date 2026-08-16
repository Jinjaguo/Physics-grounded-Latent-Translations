# EXP_R3 interface audit

EXP_R3 uses the exact released 32-D action latent and 16-frame decoder interface. F1/F2 and the decoder are loaded by their historical checkpoint paths and frozen. The source data are complete official CALVIN episode rows, not concatenated Wave27 windows. Oracle annotation boundaries provide F3 switching; no learned completion or return is evaluated.
