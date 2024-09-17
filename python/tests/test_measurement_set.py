#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import leap

MWA_MS = "../testdata/mwa/1197638568-split.ms"

def test_tensor():
    array = np.array(leap.Tensor3d(1,2,3), order = 'F', copy = False)
    array[:] = 0
    assert array.shape == (1,2,3)

def test_readcoords():
    ms = leap.MeasurementSet(MWA_MS)
    coords = np.array(ms.read_coords(0,1), order = 'F', copy = False)
    assert coords.shape == (3,5253,1)
    assert coords.flags.f_contiguous == True
    assert coords.flags.owndata == False # owndata==False is preferred
    coords = np.array(ms.read_coords(0,2), order = 'F', copy = False)
    assert coords.shape == (3,5253,2)
    assert coords.flags.f_contiguous == True
    assert coords.flags.owndata == False # owndata==False is preferred

    TOLERANCE = 0.00000001
    assert coords[0][0][0]    == pytest.approx(0,                   TOLERANCE)
    assert coords[1][0][0]    == pytest.approx(0,                   TOLERANCE)
    assert coords[2][0][0]    == pytest.approx(0,                   TOLERANCE)
    assert coords[0][1][0]    == pytest.approx(-213.234574834057,   TOLERANCE)
    assert coords[1][1][0]    == pytest.approx( 135.473926784922,   TOLERANCE)
    assert coords[2][1][0]    == pytest.approx( 136.990822255294,   TOLERANCE)
    assert coords[0][2][0]    == pytest.approx(-126.130233053304,   TOLERANCE)
    assert coords[1][2][0]    == pytest.approx( 169.064851738458,   TOLERANCE)
    assert coords[2][2][0]    == pytest.approx( 139.291586460673,   TOLERANCE)
    assert coords[0][5251][0] == pytest.approx(-366.52924769051333, TOLERANCE)
    assert coords[1][5251][0] == pytest.approx(-437.91497202478854, TOLERANCE)
    assert coords[2][5251][0] == pytest.approx(-207.55869675563417, TOLERANCE)
    assert coords[0][5252][0] == pytest.approx(0,                   TOLERANCE)
    assert coords[1][5252][0] == pytest.approx(0,                   TOLERANCE)
    assert coords[2][5252][0] == pytest.approx(0,                   TOLERANCE)
    assert coords[0][0][1]    == pytest.approx(0,                   TOLERANCE)
    assert coords[1][0][1]    == pytest.approx(0,                   TOLERANCE)
    assert coords[2][0][1]    == pytest.approx(0,                   TOLERANCE)
    assert coords[0][1][1]    == pytest.approx(-213.16346997196314, TOLERANCE)
    assert coords[1][1][1]    == pytest.approx( 135.46083100163386, TOLERANCE)
    assert coords[2][1][1]    == pytest.approx( 137.11437728855378, TOLERANCE)

@pytest.mark.skip(reason="less preferred, ocassionally contains corrupt data")
def test_readcoords_numpy():
    ms = leap.MeasurementSet(MWA_MS)
    coords = ms.read_coords(0,1).numpy_view
    assert coords.shape == (3,5253,1)
    assert coords.flags.f_contiguous == True
    assert coords.flags.owndata == False
    coords = ms.read_coords(0,2).numpy_view
    assert coords.shape == (3,5253,2)
    assert coords.flags.f_contiguous == True
    assert coords.flags.owndata == False
    
    TOLERANCE = 0.00000001
    assert coords[0][0][0]    == pytest.approx(0,                   TOLERANCE)
    assert coords[1][0][0]    == pytest.approx(0,                   TOLERANCE)
    assert coords[2][0][0]    == pytest.approx(0,                   TOLERANCE)
    assert coords[0][1][0]    == pytest.approx(-213.234574834057,   TOLERANCE)
    assert coords[1][1][0]    == pytest.approx( 135.473926784922,   TOLERANCE)
    assert coords[2][1][0]    == pytest.approx( 136.990822255294,   TOLERANCE)
    assert coords[0][2][0]    == pytest.approx(-126.130233053304,   TOLERANCE)
    assert coords[1][2][0]    == pytest.approx( 169.064851738458,   TOLERANCE)
    assert coords[2][2][0]    == pytest.approx( 139.291586460673,   TOLERANCE)
    assert coords[0][5251][0] == pytest.approx(-366.52924769051333, TOLERANCE)
    assert coords[1][5251][0] == pytest.approx(-437.91497202478854, TOLERANCE)
    assert coords[2][5251][0] == pytest.approx(-207.55869675563417, TOLERANCE)
    assert coords[0][5252][0] == pytest.approx(0,                   TOLERANCE)
    assert coords[1][5252][0] == pytest.approx(0,                   TOLERANCE)
    assert coords[2][5252][0] == pytest.approx(0,                   TOLERANCE)
    assert coords[0][0][1]    == pytest.approx(0,                   TOLERANCE)
    assert coords[1][0][1]    == pytest.approx(0,                   TOLERANCE)
    assert coords[2][0][1]    == pytest.approx(0,                   TOLERANCE)
    assert coords[0][1][1]    == pytest.approx(-213.16346997196314, TOLERANCE)
    assert coords[1][1][1]    == pytest.approx( 135.46083100163386, TOLERANCE)
    assert coords[2][1][1]    == pytest.approx( 137.11437728855378, TOLERANCE)

def test_readvis():
    ms = leap.MeasurementSet(MWA_MS)
    vis = np.array(ms.read_vis(0,1), order = 'F', copy = False)
    assert vis.shape == (4,48,5253,1)
    assert vis.flags.f_contiguous == True
    assert vis.flags.owndata == False

    # same as MeasurementSetTests.cc
    assert vis[0][0][0][0] == 0
    assert vis[1][0][0][0] == 0
    assert vis[2][0][0][0] == 0
    assert vis[3][0][0][0] == 0
    assert vis[0][0][1][0] == pytest.approx(-0.703454494476318 + -24.7045249938965j)
    assert vis[1][0][1][0] == pytest.approx(5.16687202453613 + -1.57053351402283j)
    assert vis[2][0][1][0] == pytest.approx(-10.9083280563354 + 11.3552942276001j)
    assert vis[3][0][1][0] == pytest.approx(-28.7867774963379 + 20.7210712432861j)

@pytest.mark.skip(reason="less preferred, ocassionally contains corrupt data")
def test_readvis_numpy():
    ms = leap.MeasurementSet(MWA_MS)
    vis = ms.read_vis(0,1).numpy_view
    assert vis.shape == (4,48,5253,1)
    assert vis.flags.f_contiguous == True
    assert vis.flags.owndata == False

    # same as MeasurementSetTests.cc
    assert vis[0][0][0][0] == 0
    assert vis[1][0][0][0] == 0
    assert vis[2][0][0][0] == 0
    assert vis[3][0][0][0] == 0
    assert vis[0][0][1][0] == pytest.approx(-0.703454494476318 + -24.7045249938965j)
    assert vis[1][0][1][0] == pytest.approx(5.16687202453613 + -1.57053351402283j)
    assert vis[2][0][1][0] == pytest.approx(-10.9083280563354 + 11.3552942276001j)
    assert vis[3][0][1][0] == pytest.approx(-28.7867774963379 + 20.7210712432861j)
