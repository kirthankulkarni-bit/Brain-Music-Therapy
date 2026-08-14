PRE-CORRECTION DATA - RETAINED FOR PROVENANCE, NOT VALID AS PILOT DATA
======================================================================

Every CSV in this directory was produced by the live pipeline while it declared
the Muse 2 sampling rate as 128 Hz. The Muse 2 streams EEG at 256 Hz.

Consequence: MNE received 640 samples and was told they spanned 5 seconds. They
spanned 2.5. Every frequency on the axis was therefore halved.

    mask used in code        frequencies actually measured
    alpha  8-13 Hz           true 16-26 Hz
    beta  13-30 Hz           true 26-60 Hz

The beta mask was inclusive at 30 Hz, and 60 Hz mains hum maps to exactly 30 Hz
under the halved axis. The dominant contributor to what these files record as
"Beta_Power" is therefore power line noise picked up by dry electrodes, not
neural beta activity.

This explains two things that were previously puzzling:
  - the resting baseline sitting near 0.4 rather than near 1.0
  - the need to retune the hysteresis thresholds to 0.35 / 0.55 to keep the
    state machine from chattering

WHAT THESE FILES MAY AND MAY NOT BE USED FOR

  MAY:  documenting the defect and its discovery; demonstrating the difference
        between pre- and post-correction feature distributions; the methods
        section's account of what went wrong and how it was found.

  MAY NOT: any reported alpha, beta, or beta/alpha value; any pilot effect size;
        any threshold or baseline calibration; any claim about participant state.

These files are kept rather than deleted deliberately. Discarding data because it
turned out to be wrong destroys the record of the error; keeping it with an honest
label preserves it.

Corrected sessions are written to sessions/ in JSONL format, and every session
manifest there records the sampling rate that was actually used, verified with
scripts/verify_sample_rate.py before collection.

Defect found and corrected: 2026-08-14.
