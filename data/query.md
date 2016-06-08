Data generated from CasJobs DR12 using
```
SELECT TOP 1000000 SpecObjID, z, zErr, psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z INTO mydb.psf3 FROM DR12.SpecPhotoAll
  WHERE zErr != 0 AND psfMag_u IS NOT NULL AND z<1 AND z>0 AND zErr<0.01 AND psfMag_u>-5 AND psfMag_g>-5 AND psfMag_r>-5 AND psfMag_i>-5 AND psfMag_z>-5;
```
