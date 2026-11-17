"""Measured cuBLASLt configurations of the single-direction D64 / D128 TriMul contractions (``_h100_bmm.contract``; hidden = D: the packaged
inference route below ``integrations.trimul_h100.ROWPAD_FROM`` and the port's row-padded form from it): key (L, a transposed (incoming), planes
per launch = D B) -> (tile id, custom option, cluster shape id) of algorithm 66 (no split-K, no workspace; outputs bitwise equal to the default
``bmm`` pick, checked on every entry). Candidates: the bit-equal configuration with the smallest worst-run ratio in two separate-process sweeps
of the heuristic candidates (rows of 7 interleaved rounds; <= 0.98 of the rc/2.5.13 pick); kept when the in-graph module replay
(TriangleMultiplication, no mask, interleaved off / control / on, 15 rounds) measured <= 0.995 of the rc/2.5.13 pick in two processes for
the key's own shape (P 64: D64 B1, P 128: D128 B1, P 256: D128 B2) and <= 1.002 for D64 B2 (which shares the P 128 keys); H100,
2026-11-17. The P 256 keys of ``LT_CONFIGS`` (measured on D256) not beaten are carried, so every call not listed runs what rc/2.5.13 ran.
Lengths not listed run the default heuristic pick."""

LT_UNI: dict = {
    (144, False, 64): (455, 3, 5),  # module 0.993
    (144, True, 64): (458, 3, 5),  # module 0.964
    (160, True, 64): (30, 2, 5),  # module 0.994
    (224, False, 64): (414, 5, 3),  # module 0.995
    (240, True, 64): (20, 3, 5),  # module 0.991
    (256, True, 64): (20, 1, 3),  # module 0.989
    (288, True, 64): (340, 3, 2),  # module 0.990
    (304, True, 64): (194, 4, 3),  # module 0.960
    (336, True, 64): (32, 6, 5),  # module 0.970
    (352, True, 64): (32, 6, 5),  # module 0.972
    (400, True, 64): (333, 3, 5),  # module 0.886
    (416, True, 64): (220, 0, 3),  # module 0.880
    (432, True, 64): (221, 6, 5),  # module 0.979
    (448, True, 64): (19, 2, 5),  # module 0.990
    (528, True, 64): (458, 3, 5),  # module 0.992
    (544, True, 64): (458, 3, 5),  # module 0.981
    (560, True, 64): (458, 3, 5),  # module 0.982
    (576, True, 64): (116, 2, 2),  # module 0.980
    (592, False, 64): (29, 3, 5),  # module 0.981
    (608, False, 64): (29, 3, 5),  # module 0.981
    (608, True, 64): (29, 3, 5),  # module 0.994
    (624, False, 64): (29, 3, 5),  # module 0.981
    (624, True, 64): (29, 3, 5),  # module 0.995
    (640, False, 64): (29, 3, 5),  # module 0.982
    (640, True, 64): (20, 3, 2),  # module 0.992
    (656, False, 64): (201, 2, 5),  # module 0.924
    (672, False, 64): (197, 2, 5),  # module 0.993
    (688, False, 64): (201, 2, 5),  # module 0.918
    (688, True, 64): (24, 7, 5),  # module 0.937
    (704, True, 64): (31, 7, 5),  # module 0.975
    (720, False, 64): (184, 2, 5),  # module 0.992
    (736, False, 64): (20, 1, 3),  # module 0.991
    (768, True, 64): (20, 1, 3),  # module 0.971
    (784, True, 64): (421, 7, 2),  # module 0.990
    (800, False, 64): (29, 3, 2),  # module 0.987
    (816, False, 64): (419, 7, 2),  # module 0.994
    (816, True, 64): (421, 7, 2),  # module 0.826
    (832, True, 64): (32, 6, 2),  # module 0.797
    (848, False, 64): (406, 3, 5),  # module 0.983
    (864, False, 64): (201, 2, 2),  # module 0.982
    (864, True, 64): (414, 7, 5),  # module 0.983
    (928, False, 64): (151, 0, 3),  # module 0.976
    (928, True, 64): (458, 3, 5),  # module 0.964
    (944, False, 64): (151, 0, 3),  # module 0.986
    (944, True, 64): (458, 3, 5),  # module 0.978
    (960, False, 64): (29, 1, 3),  # module 0.985
    (960, True, 64): (29, 1, 3),  # module 0.973
    (992, False, 64): (184, 2, 5),  # module 0.980
    (1008, False, 64): (184, 2, 5),  # module 0.988
    (1024, False, 64): (20, 1, 3),  # module 0.974
    (1024, True, 64): (20, 1, 3),  # module 0.974
    (1040, False, 64): (201, 0, 3),  # module 0.978
    (1056, False, 64): (201, 0, 3),  # module 0.916
    (1088, False, 64): (29, 3, 2),  # module 0.962
    (1104, False, 64): (460, 3, 5),  # module 0.960
    (1120, False, 64): (29, 3, 2),  # module 0.990
    (1120, True, 64): (414, 7, 2),  # module 0.993
    (1136, False, 64): (184, 0, 3),  # module 0.993
    (1152, False, 64): (184, 0, 3),  # module 0.965
    (1168, False, 64): (197, 2, 5),  # module 0.982
    (1184, False, 64): (201, 2, 5),  # module 0.973
    (1200, False, 64): (201, 2, 5),  # module 0.956
    (1200, True, 64): (29, 1, 3),  # module 0.995
    (1216, True, 64): (29, 1, 3),  # module 0.972
    (1232, False, 64): (201, 2, 5),  # module 0.934
    (1248, False, 64): (30, 2, 5),  # module 0.957
    (1248, True, 64): (29, 1, 3),  # module 0.976
    (1264, False, 64): (30, 2, 5),  # module 0.955
    (1264, True, 64): (29, 1, 3),  # module 0.969
    (1280, False, 64): (29, 1, 3),  # module 0.949
    (1280, True, 64): (20, 1, 3),  # module 0.959
    (1296, False, 64): (460, 3, 5),  # module 0.967
    (1312, False, 64): (151, 0, 3),  # module 0.952
    (1312, True, 64): (151, 0, 3),  # module 0.976
    (1328, False, 64): (166, 0, 3),  # module 0.966
    (1344, False, 64): (197, 0, 3),  # module 0.989
    (1360, False, 64): (409, 3, 5),  # module 0.994
    (1376, False, 64): (201, 0, 3),  # module 0.936
    (1392, False, 64): (409, 3, 5),  # module 0.951
    (1408, True, 64): (20, 3, 2),  # module 0.892
    (1456, False, 64): (412, 5, 3),  # module 0.978
    (1456, True, 64): (24, 7, 5),  # module 0.981
    (1472, False, 64): (197, 2, 5),  # module 0.951
    (1488, False, 64): (197, 2, 5),  # module 0.971
    (1488, True, 64): (24, 7, 5),  # module 0.977
    (1504, False, 64): (197, 2, 5),  # module 0.951
    (1504, True, 64): (24, 7, 5),  # module 0.984
    (1520, False, 64): (24, 7, 5),  # module 0.985
    (1520, True, 64): (24, 7, 5),  # module 0.982
    (1536, False, 64): (201, 2, 5),  # module 0.969
    (1536, True, 64): (20, 1, 3),  # module 0.949
    (1600, False, 64): (29, 3, 5),  # module 0.952
    (1632, False, 64): (197, 0, 3),  # module 0.978
    (1648, False, 64): (215, 4, 3),  # module 0.976
    (1728, False, 64): (184, 2, 5),  # module 0.986
    (1760, False, 64): (201, 2, 5),  # module 0.956
    (1760, True, 64): (29, 1, 3),  # module 0.963
    (1920, False, 64): (29, 3, 5),  # module 0.965
    (1936, False, 64): (421, 5, 3),  # module 0.991
    (1984, False, 64): (197, 2, 5),  # module 0.976
    (2048, False, 64): (29, 1, 3),  # module 0.967
    (176, True, 128): (458, 3, 5),  # module 0.987, D64 B2 0.976
    (192, False, 128): (458, 3, 5),  # module 0.993, D64 B2 0.982
    (192, True, 128): (458, 3, 5),  # module 0.984, D64 B2 0.983
    (288, True, 128): (340, 3, 2),  # module 0.995, D64 B2 0.998
    (336, False, 128): (166, 2, 5),  # module 0.985, D64 B2 0.989
    (336, True, 128): (32, 6, 5),  # module 0.968, D64 B2 0.966
    (352, True, 128): (32, 6, 5),  # module 0.970, D64 B2 0.973
    (400, False, 128): (184, 2, 5),  # module 0.968, D64 B2 0.976
    (400, True, 128): (495, 5, 3),  # module 0.916, D64 B2 0.921
    (416, False, 128): (333, 3, 5),  # module 0.985, D64 B2 0.988
    (416, True, 128): (220, 0, 3),  # module 0.887, D64 B2 0.906
    (432, False, 128): (184, 2, 5),  # module 0.962, D64 B2 0.961
    (432, True, 128): (414, 5, 3),  # module 0.989, D64 B2 0.986
    (448, True, 128): (495, 5, 3),  # module 0.992, D64 B2 0.986
    (528, True, 128): (458, 3, 5),  # module 0.989, D64 B2 0.998
    (544, True, 128): (458, 3, 5),  # module 0.989, D64 B2 0.998
    (560, False, 128): (151, 0, 3),  # module 0.991, D64 B2 0.993
    (576, True, 128): (116, 2, 2),  # module 0.984, D64 B2 0.991
    (592, False, 128): (29, 3, 5),  # module 0.967, D64 B2 0.976
    (608, False, 128): (29, 3, 5),  # module 0.959, D64 B2 0.971
    (608, True, 128): (29, 3, 5),  # module 0.969, D64 B2 0.975
    (624, False, 128): (30, 0, 3),  # module 0.985, D64 B2 0.990
    (640, False, 128): (29, 3, 5),  # module 0.963, D64 B2 0.974
    (640, True, 128): (20, 3, 2),  # module 0.975, D64 B2 0.979
    (656, False, 128): (197, 2, 5),  # module 0.915, D64 B2 0.911
    (672, False, 128): (197, 2, 5),  # module 0.965, D64 B2 0.980
    (688, False, 128): (201, 2, 5),  # module 0.926, D64 B2 0.915
    (688, True, 128): (24, 7, 5),  # module 0.958, D64 B2 0.933
    (720, False, 128): (184, 2, 5),  # module 0.956, D64 B2 0.973
    (736, False, 128): (20, 1, 3),  # module 0.966, D64 B2 0.978
    (736, True, 128): (151, 0, 3),  # module 0.968, D64 B2 0.992
    (768, False, 128): (20, 1, 3),  # module 0.976, D64 B2 0.993
    (768, True, 128): (20, 1, 3),  # module 0.963, D64 B2 0.964
    (784, True, 128): (421, 7, 2),  # module 0.989, D64 B2 0.965
    (800, False, 128): (29, 3, 2),  # module 0.981, D64 B2 0.978
    (832, True, 128): (20, 3, 2),  # module 0.794, D64 B2 0.829
    (848, False, 128): (406, 3, 5),  # module 0.942, D64 B2 0.994
    (864, False, 128): (184, 0, 3),  # module 0.960, D64 B2 0.974
    (880, False, 128): (409, 3, 2),  # module 0.975, D64 B2 0.978
    (896, False, 128): (165, 0, 3),  # module 0.940, D64 B2 0.945
    (896, True, 128): (20, 3, 2),  # module 0.955, D64 B2 0.980
    (912, False, 128): (458, 3, 5),  # module 0.962, D64 B2 0.993
    (912, True, 128): (458, 3, 5),  # module 0.976, D64 B2 1.000
    (928, False, 128): (151, 0, 3),  # module 0.957, D64 B2 0.977
    (928, True, 128): (458, 3, 5),  # module 0.949, D64 B2 0.973
    (944, False, 128): (458, 3, 5),  # module 0.960, D64 B2 0.982
    (960, False, 128): (29, 1, 3),  # module 0.981, D64 B2 0.988
    (960, True, 128): (29, 1, 3),  # module 0.966, D64 B2 0.973
    (976, False, 128): (197, 2, 5),  # module 0.952, D64 B2 0.978
    (992, False, 128): (197, 2, 5),  # module 0.959, D64 B2 0.972
    (1008, False, 128): (197, 2, 5),  # module 0.949, D64 B2 0.965
    (1024, False, 128): (20, 1, 3),  # module 0.958, D64 B2 0.975
    (1024, True, 128): (20, 1, 3),  # module 0.959, D64 B2 0.970
    (1040, False, 128): (201, 0, 3),  # module 0.943, D64 B2 0.978
    (1040, True, 128): (32, 6, 5),  # module 0.976, D64 B2 0.956
    (1056, False, 128): (201, 0, 3),  # module 0.912, D64 B2 0.915
    (1088, False, 128): (29, 3, 2),  # module 0.989, D64 B2 0.999
    (1152, False, 128): (184, 0, 3),  # module 0.963, D64 B2 0.972
    (1184, False, 128): (201, 2, 5),  # module 0.968, D64 B2 0.948
    (1200, False, 128): (201, 2, 5),  # module 0.956, D64 B2 0.958
    (1216, False, 128): (201, 2, 5),  # module 0.937, D64 B2 0.943
    (1216, True, 128): (29, 1, 3),  # module 0.984, D64 B2 0.955
    (1232, False, 128): (409, 1, 3),  # module 0.969, D64 B2 0.966
    (1248, False, 128): (30, 2, 5),  # module 0.941, D64 B2 0.963
    (1248, True, 128): (29, 1, 3),  # module 0.976, D64 B2 0.988
    (1264, False, 128): (30, 2, 5),  # module 0.983, D64 B2 0.961
    (1280, False, 128): (29, 1, 3),  # module 0.955, D64 B2 0.957
    (1280, True, 128): (29, 1, 3),  # module 0.965, D64 B2 0.958
    (1296, False, 128): (460, 3, 5),  # module 0.949, D64 B2 0.961
    (1328, False, 128): (460, 3, 5),  # module 0.984, D64 B2 0.957
    (1360, False, 128): (409, 3, 5),  # module 0.991, D64 B2 0.960
    (1376, False, 128): (201, 0, 3),  # module 0.960, D64 B2 0.946
    (1392, False, 128): (409, 3, 5),  # module 0.942, D64 B2 0.961
    (1408, True, 128): (20, 3, 2),  # module 0.817, D64 B2 0.818
    (1456, True, 128): (24, 7, 5),  # module 0.987, D64 B2 0.991
    (1472, False, 128): (197, 2, 5),  # module 0.941, D64 B2 0.950
    (1472, True, 128): (495, 7, 5),  # module 0.994, D64 B2 0.994
    (1488, False, 128): (197, 2, 5),  # module 0.954, D64 B2 0.973
    (1488, True, 128): (24, 7, 5),  # module 0.987, D64 B2 0.988
    (1504, False, 128): (197, 2, 5),  # module 0.942, D64 B2 0.964
    (1520, True, 128): (24, 7, 5),  # module 0.985, D64 B2 0.986
    (1536, False, 128): (201, 2, 5),  # module 0.970, D64 B2 0.961
    (1600, False, 128): (29, 3, 5),  # module 0.954, D64 B2 0.953
    (1664, False, 128): (197, 0, 3),  # module 0.933, D64 B2 0.937
    (1680, False, 128): (227, 6, 5),  # module 0.986, D64 B2 0.990
    (1760, False, 128): (201, 2, 5),  # module 0.958, D64 B2 0.954
    (1760, True, 128): (29, 1, 3),  # module 0.966, D64 B2 0.999
    (1872, False, 128): (470, 5, 3),  # module 0.978, D64 B2 0.992
    (1920, False, 128): (29, 3, 5),  # module 0.975, D64 B2 0.968
    (1936, False, 128): (421, 5, 3),  # module 0.989, D64 B2 1.000
    (1984, False, 128): (197, 2, 5),  # module 0.993, D64 B2 0.983
    (2048, False, 128): (29, 1, 3),  # module 0.978, D64 B2 0.988
    (256, False, 256): (20, 1, 3),  # LT_CONFIGS (rc/2.5.13)
    (256, True, 256): (20, 1, 3),  # LT_CONFIGS (rc/2.5.13)
    (336, False, 256): (166, 2, 5),  # module 0.979
    (336, True, 256): (32, 6, 5),  # module 0.971
    (352, False, 256): (255, 2, 5),  # module 0.987
    (352, True, 256): (32, 6, 5),  # module 0.976
    (400, False, 256): (330, 3, 5),  # module 0.987
    (400, True, 256): (333, 3, 5),  # module 0.925
    (416, False, 256): (333, 3, 5),  # module 0.980
    (416, True, 256): (220, 0, 3),  # module 0.892
    (432, False, 256): (333, 3, 5),  # module 0.956
    (512, False, 256): (20, 1, 3),  # LT_CONFIGS (rc/2.5.13)
    (512, True, 256): (20, 1, 3),  # LT_CONFIGS (rc/2.5.13)
    (528, True, 256): (458, 3, 5),  # module 0.976
    (544, False, 256): (151, 0, 3),  # module 0.993
    (544, True, 256): (151, 0, 3),  # module 0.973
    (560, False, 256): (151, 0, 3),  # module 0.992
    (560, True, 256): (458, 3, 5),  # module 0.992
    (576, True, 256): (116, 2, 2),  # module 0.984
    (592, False, 256): (29, 3, 5),  # module 0.963
    (592, True, 256): (29, 3, 5),  # module 0.972
    (608, False, 256): (29, 3, 5),  # module 0.959
    (608, True, 256): (29, 3, 5),  # module 0.967
    (624, False, 256): (29, 3, 5),  # module 0.984
    (640, False, 256): (29, 3, 5),  # LT_CONFIGS (rc/2.5.13)
    (640, True, 256): (29, 3, 5),  # LT_CONFIGS (rc/2.5.13)
    (656, False, 256): (197, 2, 5),  # module 0.898
    (672, False, 256): (197, 2, 5),  # module 0.976
    (688, True, 256): (24, 7, 5),  # module 0.944
    (704, True, 256): (24, 7, 5),  # module 0.970
    (720, False, 256): (184, 2, 5),  # module 0.971
    (736, False, 256): (151, 0, 3),  # module 0.982
    (736, True, 256): (151, 0, 3),  # module 0.982
    (752, False, 256): (151, 0, 3),  # module 0.984
    (768, False, 256): (20, 1, 3),  # LT_CONFIGS (rc/2.5.13)
    (768, True, 256): (20, 1, 3),  # LT_CONFIGS (rc/2.5.13)
    (784, True, 256): (421, 7, 2),  # module 0.973
    (832, True, 256): (248, 5, 2),  # module 0.745
    (848, False, 256): (406, 3, 5),  # module 0.977
    (880, False, 256): (409, 3, 2),  # module 0.968
    (896, False, 256): (165, 0, 3),  # module 0.927
    (896, True, 256): (29, 3, 5),  # LT_CONFIGS (rc/2.5.13)
    (912, False, 256): (458, 3, 5),  # module 0.980
    (928, False, 256): (458, 3, 5),  # module 0.959
    (928, True, 256): (458, 3, 5),  # module 0.959
    (960, False, 256): (29, 1, 3),  # module 0.979
    (992, False, 256): (184, 2, 5),  # module 0.986
    (1024, False, 256): (20, 1, 3),  # LT_CONFIGS (rc/2.5.13)
    (1024, True, 256): (20, 1, 3),  # LT_CONFIGS (rc/2.5.13)
    (1040, False, 256): (201, 0, 3),  # module 0.954
    (1040, True, 256): (32, 6, 5),  # module 0.986
    (1056, False, 256): (201, 0, 3),  # module 0.892
    (1104, False, 256): (460, 3, 5),  # module 0.976
    (1152, False, 256): (184, 0, 3),  # module 0.964
    (1184, False, 256): (201, 2, 5),  # module 0.973
    (1200, False, 256): (227, 6, 5),  # module 0.983
    (1216, False, 256): (201, 2, 5),  # module 0.937
    (1216, True, 256): (23, 4, 3),  # module 0.995
    (1232, True, 256): (24, 7, 5),  # module 0.989
    (1280, False, 256): (29, 1, 3),  # LT_CONFIGS (rc/2.5.13)
    (1280, True, 256): (24, 7, 5),  # module 0.995
    (1296, False, 256): (460, 3, 5),  # module 0.971
    (1296, True, 256): (209, 4, 2),  # module 0.983
    (1312, True, 256): (209, 4, 2),  # module 0.993
    (1328, False, 256): (166, 0, 3),  # module 0.980
    (1360, True, 256): (421, 7, 2),  # module 0.990
    (1376, False, 256): (201, 0, 3),  # module 0.941
    (1392, True, 256): (421, 7, 2),  # module 0.992
    (1408, True, 256): (24, 5, 3),  # module 0.792
    (1456, False, 256): (470, 5, 3),  # module 0.986
    (1472, False, 256): (197, 2, 5),  # module 0.965
    (1504, False, 256): (197, 2, 5),  # module 0.975
    (1504, True, 256): (24, 7, 5),  # module 0.989
    (1536, False, 256): (201, 2, 5),  # LT_CONFIGS (rc/2.5.13)
    (1536, True, 256): (24, 7, 5),  # module 0.984
    (1552, False, 256): (419, 7, 5),  # module 0.993
    (1552, True, 256): (465, 7, 5),  # module 0.981
    (1568, False, 256): (465, 7, 5),  # module 0.975
    (1568, True, 256): (465, 7, 5),  # module 0.982
    (1584, False, 256): (194, 4, 3),  # module 0.994
    (1584, True, 256): (465, 7, 5),  # module 0.973
    (1600, False, 256): (29, 3, 5),  # module 0.959
    (1648, False, 256): (470, 7, 5),  # module 0.992
    (1680, False, 256): (463, 7, 5),  # module 0.978
    (1680, True, 256): (421, 5, 3),  # module 0.980
    (1728, False, 256): (201, 2, 5),  # module 0.988
    (1744, True, 256): (414, 5, 3),  # module 0.986
    (1760, False, 256): (201, 2, 5),  # module 0.956
    (1920, False, 256): (29, 3, 5),  # module 0.966
    (1936, False, 256): (500, 7, 5),  # module 0.857
    (2560, False, 256): (29, 1, 3),  # LT_CONFIGS (rc/2.5.13)
}
