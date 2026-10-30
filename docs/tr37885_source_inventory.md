# TR 37.885 source inventory

The implementation baseline is 3GPP TR 37.885 V15.3.0 (Release 15), including
CR 0006. The shadow-fading procedures referenced by the report are traced to
TR 36.885 V14.0.0, and reused coefficient procedures are limited to the
applicable Release-15 TR 38.901 semantics.

Packaged resources under
`channel_dataset/resources/standards/wireless/tr_37_885/v15_3` record the
source revision, clause or table, units, canonical checksum, transcription
date, and review status for:

- the Annex A reference urban grid, vehicles, and mobility Options A/B;
- urban LOS/NLOSv probability, path loss, shadowing, and blockage parameters;
- urban LOS, NLOSv, and NLOS fixed CDL profiles; and
- the 6 GHz vehicle-antenna evaluation assumptions.

TR 38.901 kernels are shared only for backend-neutral mechanics such as
canonical CIR storage and response conversion. TR 37.885 tables and
state-dependent rules remain in their own namespace. All new transcriptions
ship as `pending_independent_review`; checksums detect changes but do not
substitute for an independent standards review.
