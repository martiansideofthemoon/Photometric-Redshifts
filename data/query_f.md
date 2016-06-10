Data generated from CasJobs DR12 using
```
SELECT TOP 1000000 specObjID, z, zErr, class, subClass, zWarning,
  calibStatus_u, calibStatus_g, calibStatus_i, calibStatus_r, calibStatus_z, 
  dered_u, dered_g, dered_i, dered_r, dered_z into mydb.Table_S from DR12.SpecPhotoAll;
```
