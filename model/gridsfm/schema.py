"""Single source of truth for column indices and edge-type keys."""
from __future__ import annotations

BUS_TYPE_IDX = 1
BUS_VMIN_IDX = 2
BUS_VMAX_IDX = 3

GEN_PMIN_IDX = 2
GEN_PMAX_IDX = 3
GEN_QMIN_IDX = 5
GEN_QMAX_IDX = 6
GEN_VG_IDX   = 7
GEN_CP2_IDX  = 8
GEN_CP1_IDX  = 9
GEN_CP0_IDX  = 10

LOAD_PD_IDX = 0
LOAD_QD_IDX = 1

SHUNT_BS_IDX = 0
SHUNT_GS_IDX = 1

AC_LINE_ANGMIN_IDX  = 0
AC_LINE_ANGMAX_IDX  = 1
AC_LINE_BFR_IDX     = 2
AC_LINE_BTO_IDX     = 3
AC_LINE_R_IDX       = 4
AC_LINE_X_IDX       = 5
AC_LINE_RATE_A_IDX  = 6

TR_ANGMIN_IDX       = 0
TR_ANGMAX_IDX       = 1
TR_R_IDX            = 2
TR_X_IDX            = 3
TR_RATE_A_IDX       = 4
TR_TAP_IDX          = 7
TR_SHIFT_IDX        = 8
TR_BFR_IDX          = 9
TR_BTO_IDX          = 10

# ON/OFF availability, APPENDED as the last raw column by the Julia exporter
# (export_gridsfm_data.jl). Indices are the last raw column of each element
# type, every index above is unchanged.
BUS_AVAIL_IDX       = 4
GEN_AVAIL_IDX       = 11
LOAD_AVAIL_IDX      = 2
SHUNT_AVAIL_IDX     = 2
AC_LINE_AVAIL_IDX   = 9
TR_AVAIL_IDX        = 11

AC_LINE_KEY     = ("bus", "ac_line",     "bus")
TRANSFORMER_KEY = ("bus", "transformer", "bus")

REACTANCE_EPS   = 1e-10
REACTANCE_FLOOR = 1e-4

PI_Z2_EPS  = 1e-12
PI_TAP_EPS = 1e-9

DC_PRIOR_SCALE_MIN = 0.5
DC_PRIOR_SCALE_MAX = 2.0
