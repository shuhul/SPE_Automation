"""
Quick test of pl_spec.pl_spec_lf — single-point scan, no stage movement.
"""
import pl_spec
import sgd

sgd.sgd_init()

pl_spec.pl_spec_lf(
    xdim=0, ydim=0, dx=0, dy=0,
    center=(0, 0),
    grating=150,
    exposure_time=1,
    center_wavelength=700,
    foldername='test_pl_spec_lf',
    scan_type='single',
    current_user='shuhul',
)
