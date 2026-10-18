# Raman processing reference

`raman_processing_reference.npz` contains X and Y arrays read from the user-provided
`RMP00084.jdx`, plus baseline-corrected and smoothed arrays captured before
extracting processing from Jdx. It makes numerical regression tests independent
of the user's untracked data directory.

Source title: RMP00084 Lead white; Forbes; MFAB; Bottle 519; 785; MFAB; scat.
Source origin header: http://www.irug.org/jcamp-details?id=3475
Original file SHA256:
`0439c6e9f2ad5e4e89f815d9eb924fc142a4704433fcda060c605a33794469c8`.

The 2,961-point input was loaded with `auto_tidy=False`, corrected using
`Jdx.baseline_correction()` defaults (`lam=2e7`, `p=0.005`, `niter=10`), then
smoothed using `Jdx.smooth()` defaults (Savitzky–Golay window 31, polynomial 3).
The reference captures the legacy numerical behavior, not a claim of scientific
ground truth. Do not regenerate it from the extracted implementation merely to
make tests pass.
