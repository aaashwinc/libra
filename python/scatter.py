import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt


data = np.array([[100.468, 129.774, 194.655], [36.0131, 103.049, 193.354], [60.0832, 145.846, 185.762], [142.43, 35.8006, 185.906], [133.163, 1.26248, 198.91], [45.0573, 66.308, 190.394], [82.7593, 81.3706, 196.443], [145.404, 93.7652, 194.424], [71.7191, 50.0115, 161.518], [113.162, 79.2539, 197.778], [143.417, 25.7021, 194.516], [87.1918, 44.2809, 192.904], [82.0649, 107.706, 194.999], [6.76537, 142.282, 185.417], [29.7992, 128.522, 195.524], [72.9506, 139.913, 189.478], [101.555, 76.051, 196.805], [144.021, 59.5617, 153.758], [103.896, 15.5099, 182.682], [81.0495, 137.057, 193.429], [71.8632, 142.537, 196.131], [122.591, 90.9008, 168.97], [118.567, 92.7303, 184.472], [113.582, 32.9276, 197.057], [66.7333, 116.909, 198.756], [121.5, 93.7425, 192.255], [48.9716, 87.8007, 187.41], [2.53351, 86.6904, 175.33], [110.512, 4.42588, 140.132], [65.1468, 65.1926, 180.69], [27.8043, 43.7589, 196.69], [133.266, 94.7316, 137.499], [134.653, 95.257, 128.86], [147.156, 83.3882, 164.77], [92.4774, 120.36, 119.312], [132.95, 64.2079, 181.613], [143.675, 14.2637, 127.681], [138.467, 22.396, 138.315], [20.0112, 102.138, 117.022], [55.9942, 5.88227, 197.64], [29.1187, 79.6717, 197.637], [125.243, 71.8027, 137.254], [83.7061, 148.451, 109.07], [145.682, 3.52507, 190.162], [12.6575, 80.8483, 186.375], [27.2928, 148.023, 128.908], [41.789, 145.057, 187.068], [102.33, 83.559, 150.733], [101.565, 108.659, 141.602], [109.234, 144.276, 117.879], [145.695, 7.87774, 115.381], [4.22313, 89.5351, 162.765], [20.9958, 131.239, 69.5024], [14.1532, 126.78, 131.761], [83.5657, 122.959, 153.965], [124.724, 121.58, 157.265], [137.242, 58.4448, 81.664], [138.75, 25.6159, 3.45375], [93.3787, 86.934, 15.2032], [49.1088, 132.405, 80.5606], [86.4072, 148.241, 3.36096], [59.0633, 114.141, 165.756], [79.694, 125.329, 129.649], [89.4291, 113.86, 126.679], [9.58285, 54.0182, 167.439], [5.73678, 72.5638, 132.751], [44.2855, 99.6439, 172.483], [30.8872, 97.2737, 92.5541], [39.5978, 134.823, 27.2726], [86.0656, 83.2128, 182.757], [88.2709, 93.2745, 172.198], [81.6435, 118.739, 188.941], [89.0347, 101.778, 161.494], [92.7363, 47.8613, 53.8018], [114.715, 141.851, 87.4005], [132.074, 144.19, 77.3332], [79.5077, 84.5013, 38.4019], [10.723, 60.1232, 76.0102], [23.9528, 101.378, 136.753], [1.26913, 86.134, 55.6266], [72.4015, 15.8064, 154.351], [143.498, 148.149, 169.854], [144.879, 53.8029, 51.8735], [144.579, 136.925, 15.2698], [143.347, 102.166, 73.8448], [130.417, 110.903, 5.52103], [130.504, 108.633, 13.0962], [72.8906, 67.0662, 127.379], [47.2189, 93.8905, 122.943], [135.607, 6.07138, 4.57165], [26.8524, 3.869, 176.398], [109.153, 106.635, 130.862], [121.599, 144.092, 128.737], [147.074, 50.1932, 144.849], [131.769, 143.856, 22.2255], [90.6933, 120.674, 26.497], [13.0356, 147.841, 97.4889], [7.15901, 43.9835, 141.728], [50.159, 46.9828, 149.267], [20.8291, 15.0779, 80.7349], [102.907, 104.732, 150.321], [115.362, 121.222, 63.9189], [148.806, 92.624, 51.9395], [92.1709, 134.749, 3.16694], [59.4187, 84.7263, 151.879], [47.6175, 147.976, 88.3764], [38.367, 130.923, 93.1402], [30.3543, 148.031, 45.9692], [111.917, 62.678, 80.8926], [21.5325, 35.837, 157.987], [69.0271, 10.9124, 142.301], [94.5024, 45.5231, 46.4412], [139.717, 116.759, 44.4639], [117.936, 122.207, 56.5997], [144.775, 53.0181, 44.5474], [125.774, 56.1717, 51.0198], [82.4634, 121.871, 58.0869], [86.6139, 86.2813, 104.179], [43.004, 62.7706, 3.70309], [132.648, 3.73527, 16.6874], [109.685, 13.8623, 28.4091], [80.8435, 18.1987, 61.4495], [88.4632, 54.5723, 11.9856], [60.4448, 35.375, 3.39716], [84.0454, 11.1997, 104.377], [71.303, 46.4635, 45.1118], [3.26162, 51.589, 91.2417], [55.4899, 77.5785, 79.1925], [56.3131, 78.7157, 85.6492], [71.7075, 59.0981, 66.5904], [74.9894, 48.361, 32.9842], [43.1931, 63.8951, 148.231], [68.2688, 9.8523, 103.18], [56.6697, 38.2232, 79.724], [33.781, 2.0492, 70.9571], [70.2421, 58.509, 78.9083], [92.2197, 87.9338, 4.68148], [54.6563, 111.209, 87.1177], [56.8572, 50.488, 93.4286], [45.3194, 18.2889, 41.6454], [78.6164, 71.0568, 11.7278], [59.8115, 35.3242, 112.97], [108.738, 77.2491, 105.083], [142.726, 59.4626, 3.89268], [7.45461, 125.464, 105.601], [135.982, 147.679, 160.577], [82.0076, 127.937, 137.882], [86.553, 99.3036, 85.0272], [40.0717, 122.046, 123.431], [29.7722, 148.691, 56.2536], [40.0403, 135.607, 37.753], [145.682, 134.615, 6.33766], [25.6924, 50.0018, 59.1239], [86.8786, 43.138, 107.181], [126.299, 65.5603, 115.195], [40.7253, 28.8663, 146.943], [105.674, 17.5377, 40.8129], [102.243, 115.245, 33.2958], [26.6108, 128.838, 184.771], [102.51, 148.697, 158.321], [122.794, 123.679, 88.3002], [86.6296, 121.949, 38.6663], [36.0163, 91.9394, 80.5208], [116.175, 102.567, 76.0546], [140.289, 25.2903, 25.5324], [145.518, 136.849, 63.8374], [48.5931, 98.055, 46.9716], [115.942, 35.5046, 24.5896], [127.115, 33.0294, 75.3039], [59.228, 38.2541, 68.3312], [108.975, 71.283, 71.3392], [145.782, 126.819, 180.345], [142.074, 147.234, 195.314], [6.11346, 125.987, 167.012], [26.5965, 99.3745, 174.999], [96.4326, 12.2785, 152.945], [24.028, 63.9337, 199.313], [131.592, 76.7243, 152.909], [78.388, 146.104, 162.406], [78.3243, 109.929, 152.884], [85.4738, 121.197, 48.1657], [56.7964, 110.29, 99.2141], [17.8429, 115.768, 24.4879], [128.42, 28.5123, 140.943], [52.8877, 23.7776, 137.325], [147.262, 101.793, 95.0251], [144.274, 0.570521, 49.7083], [148.878, 147.889, 71.884], [121.127, 111.081, 196.63], [56.9628, 74.0693, 176.642], [86.2014, 85.8233, 189.253], [49.1413, 138.256, 196.069], [33.5477, 67.5298, 177.499], [126.408, 56.2856, 190.898], [147.665, 9.49176, 152.357], [5.15456, 113.854, 188.931], [62.6655, 119.021, 146.818], [139.376, 58.9312, 63.1015], [123.913, 131.463, 172.446], [111.93, 67.2542, 95.958], [105.397, 145.443, 24.0139], [140.456, 143.313, 29.7828], [95.5737, 113.017, 1.82481], [22.3822, 57.5619, 129.451], [45.6756, 32.9262, 161.703], [92.0706, 46.2922, 59.2931], [144.505, 138.164, 21.2172], [74.9913, 25.8071, 101.493], [53.6104, 74.4015, 1.18604], [140.707, 117.86, 121.112], [102.012, 131.162, 101.089], [3.40608, 57.576, 102.245], [55.0263, 84.2492, 46.2781], [21.0884, 148.279, 139.241], [1.96857, 76.897, 156.043], [107.023, 101.287, 90.3844], [24.1639, 141.884, 176.444], [94.6329, 70.8516, 167.524], [130.01, 72.7082, 195.673], [68.7348, 90.2358, 180.159], [0.98397, 9.80985, 195.538], [144.216, 79.2865, 109.373], [62.229, 27.0001, 196.681], [98.7758, 139.891, 183.498], [2.66649, 136.379, 141.812], [16.1164, 117.959, 33.86], [131.318, 117.971, 143.413], [1.8925, 45.3211, 181.112], [131.701, 34.4423, 128.515], [106.401, 102.031, 68.2756], [48.3958, 146.268, 67.1869], [60.9972, 79.2793, 139.357], [64.1391, 138.518, 44.3368], [98.8792, 145.534, 70.606], [125.597, 14.5312, 48.9778], [55.4273, 79.9517, 71.8288], [134.699, 77.9362, 47.2708], [33.2165, 81.6445, 91.6118], [107.315, 50.6263, 197.704], [78.5603, 25.3131, 196.838], [148.252, 119.927, 190.963], [34.1467, 27.4363, 189.583], [137.653, 0.846315, 177.358], [6.91511, 132.236, 125.649], [92.1186, 146.231, 82.1005], [81.1305, 124.642, 69.3187], [53.6785, 130.993, 65.6708], [5.64428, 100.475, 140.95], [6.15792, 130.342, 118.613], [133.655, 144.845, 10.4085], [130.938, 98.6103, 161.613], [90.9353, 118.797, 12.4964], [57.829, 37.6026, 11.1013], [77.226, 27.0633, 104.966], [132.962, 142.482, 16.5435], [21.497, 73.0708, 134.663], [135.854, 25.9307, 36.4891], [102.097, 84.196, 73.8746], [72.01, 8.64074, 28.9894], [144.228, 0.701314, 57.3106], [139.455, 54.8881, 97.4686], [145.462, 139.274, 118.315], [31.441, 80.7392, 120.326], [94.1745, 51.656, 3.84688], [1.99283, 57.1097, 198.543], [137.874, 83.3303, 194.981], [147.665, 78.5287, 150.263], [76.6289, 148.622, 139.659], [90.1069, 3.60355, 179.946], [98.4672, 4.1858, 166.618], [58.0824, 146.335, 122.522], [13.3, 123.32, 120.691], [73.3102, 118.822, 114.6], [103.331, 117.178, 102.108], [35.6258, 102.625, 159.665], [124.426, 20.2322, 84.0046], [7.21084, 102.092, 149.698], [100.25, 73.1422, 178.375], [53.3241, 5.03161, 143.25], [77.1041, 23.1076, 120.114], [144.361, 101.319, 84.2965], [144.845, 121.325, 67.8601], [134.54, 94.0873, 58.5407], [125.705, 54.7767, 41.5445], [146.52, 30.9802, 96.09], [35.2865, 90.1676, 101.993], [142.302, 95.6717, 13.7733], [26.7691, 127.091, 78.0533], [71.9715, 12.9504, 95.244], [84.0346, 140.982, 99.7293], [42.6067, 39.739, 129.212], [24.8998, 94.4126, 128.467], [139.403, 56.321, 88.7298], [132.447, 95.0364, 67.4959], [126.9, 48.0154, 170.28], [100.791, 33.7597, 16.4335], [85.6853, 148.737, 117.791], [146.436, 127.085, 171.779], [108.944, 74.9484, 60.5512], [50.526, 102.123, 28.6243], [68.4094, 37.7254, 187.127], [147.698, 0.876451, 20.0992], [104.896, 35.0003, 4.12237], [19.6342, 113.798, 16.7598], [123.538, 1.38001, 123.379], [147.49, 22.3916, 49.245], [60.7839, 144.622, 4.10968], [145.196, 24.2425, 39.1137], [12.6541, 146.321, 88.0073], [30.7839, 14.5084, 196.111], [55.485, 123.845, 198.271], [136.82, 83.9772, 94.1329], [85.7461, 68.2465, 150.521], [104.043, 88.7088, 135.915], [21.0445, 70.1087, 165.279], [111.754, 57.7363, 75.3114], [11.1169, 59.1897, 67.2212], [87.9813, 7.53117, 84.974], [68.0549, 2.86513, 123.028], [107.707, 147.732, 16.5599], [69.9524, 68.8874, 34.1152], [77.8951, 3.5254, 58.7516], [85.1734, 67.7453, 1.02407], [66.5367, 33.3968, 87.6753], [21.9726, 16.3313, 90.6454], [111.775, 101.475, 80.4063], [3.69178, 55.0439, 175.039], [10.6614, 56.7074, 55.1661], [31.1358, 1.22015, 134.632], [96.2734, 22.6044, 192.745], [98.2749, 96.6113, 118.245], [123.571, 34.5513, 90.0503], [146.116, 130.087, 192.933], [83.7518, 35.2906, 146.468], [1.3128, 91.574, 196.871], [8.8873, 112.631, 197.25], [45.7579, 122.176, 104.417], [50.168, 147.026, 83.3445], [136.608, 108.423, 196.09], [128.775, 17.0984, 60.954], [146.241, 116.112, 109.953], [60.9051, 82.1369, 118.552], [1.73575, 54.0859, 113.851], [85.6, 100.825, 75.7354], [84.4363, 42.5147, 98.2119], [40.9352, 62.5471, 13.3928], [41.0864, 138.58, 49.9099], [22.2101, 99.307, 133.169], [145.943, 138.816, 110.353], [57.7495, 94.7586, 161.95], [56.6075, 112.575, 156.1], [148.531, 43.2252, 198.585], [104.027, 133.839, 150.163], [120.229, 13.8737, 96.8999], [137.529, 60.0503, 69.9207], [94.415, 40.2164, 125.703], [148.557, 118.643, 139.234], [130.938, 117.262, 136.078], [65.4056, 134.492, 31.6671], [132.575, 144.246, 84.7576], [52.2668, 134.973, 11.1856], [103.628, 1.14941, 105.645], [94.9957, 44.4664, 36.3888], [70.801, 33.6272, 77.3214], [44.9529, 16.0223, 52.6547], [141.328, 28.7679, 8.76815], [24.5943, 77.2567, 159.224], [52.8679, 52.6637, 113.296], [92.1646, 35.3533, 54.57], [4.41967, 125.923, 157.294], [115.73, 36.4936, 174.048], [82.9219, 121.564, 148.542], [54.1696, 81.6795, 105.868], [80.8555, 126.298, 79.4309], [137.646, 18.2292, 172.517], [82.1405, 122.582, 89.4029], [16.6654, 128.735, 140.54], [22.5417, 119.355, 147.919], [138.168, 60.3574, 75.6796], [127.984, 58.3149, 13.4483], [71.0386, 11.5037, 40.611], [61.446, 32.7124, 101.034], [75.4935, 26.2692, 25.5415], [113.626, 41.1314, 184.272], [92.4005, 122.992, 111.369], [114.388, 34.2546, 14.1483], [38.8968, 72.5331, 121.296], [127.324, 56.9933, 20.6654], [108.528, 12.8279, 16.5975], [51.1899, 38.2789, 172.615], [124.884, 4.39585, 77.3044], [31.7893, 2.01507, 123.702], [1.01666, 91.0338, 191.678], [49.4215, 145.976, 74.9689], [70.1208, 67.2918, 43.8456], [11.7644, 147.633, 198.863], [145.822, 124.023, 78.0387], [30.862, 120.975, 171.502], [44.4026, 24.4302, 35.1606], [20.5985, 84.9, 37.5867], [126.945, 92.429, 113.393], [104.088, 116.571, 112.889], [137.399, 96.0615, 4.36911], [5.06848, 148.51, 104.75], [26.8047, 38.5132, 147.308], [136.445, 76.564, 34.42], [138.908, 68.434, 45.5396], [139.754, 118.601, 33.2343], [128.011, 112.056, 32.6032], [73.2412, 69.9638, 27.1201], [88.7164, 71.5291, 157.77], [103.121, 6.38333, 71.8043], [15.8603, 39.0169, 186.408], [15.7506, 63.0125, 154.605], [97.7457, 84.8948, 25.6131], [1.14122, 87.3862, 48.1075], [19.4781, 86.3103, 46.9633], [78.5724, 2.57113, 50.6305], [107.945, 15.7261, 192.944], [21.3298, 104.655, 126.363], [97.8883, 140.655, 49.4363], [20.4462, 87.3477, 32.0872], [143.986, 0.529721, 46.5341], [120.03, 121.622, 49.2114], [120.191, 18.2116, 107.21], [141.984, 146.585, 186.603], [100.318, 118.316, 184.117], [148.133, 24.1413, 66.2516], [14.3351, 37.1988, 174.955], [30.5631, 31.0491, 177.689], [21.9812, 131.883, 63.6154], [46.8459, 22.2582, 154.29], [126.334, 65.6847, 109.424], [147.747, 61.9647, 111.764], [44.0113, 119.604, 111.915], [148.164, 100.408, 170.53], [35.3205, 147.95, 38.7079], [64.579, 41.5764, 54.2227], [144.071, 90.3623, 26.8387], [127.63, 19.3433, 70.8141], [83.6298, 23.0726, 81.2154], [52.7892, 53.6999, 46.6328], [57.5313, 84.367, 39.0555], [141.741, 0.197082, 40.7854], [146.943, 72.8833, 94.0825], [146.624, 137.93, 103.156], [13.1626, 144.919, 81.3633], [39.5221, 129.972, 86.7154], [91.5077, 142.331, 149.388], [94.026, 60.8901, 109.212], [124.748, 53.5437, 33.1328], [103.323, 0.845254, 100.596], [97.4483, 0.928431, 91.9274], [32.0464, 2.24561, 116.553], [40.3023, 63.9832, 159.105], [68.7933, 14.3381, 48.7791], [83.0597, 118.296, 95.749], [25.8852, 102.62, 142.545], [75.742, 29.1665, 40.2614], [126.061, 64.2964, 128.871], [134.999, 146.365, 144.901], [98.8447, 84.8007, 33.6509], [96.7182, 140.206, 56.0215], [116.006, 122.67, 70.2665], [26.6227, 1.04701, 143.169], [47.3354, 148.457, 95.8418], [35.5349, 148.251, 99.0517], [87.7565, 86.6053, 96.946], [29.3637, 111.703, 72.2908], [50.2401, 92.1161, 56.0043], [53.8331, 79.3743, 93.2074], [88.5198, 50.368, 23.8356], [37.8089, 63.0734, 23.9971], [105.722, 100.904, 98.0657], [34.1321, 136.745, 107.61], [147.208, 121.494, 91.0658], [146.967, 136.253, 95.9374], [126.543, 2.39682, 26.9128], [14.753, 148.503, 152.023], [7.13196, 90.0801, 147.243], [123.725, 101.401, 91.6594], [117.755, 66.1965, 199.782], [120.826, 140.167, 188.906], [41.4697, 137.004, 58.7313], [98.4066, 18.4444, 47.2505], [105.639, 8.33217, 52.7133], [145.637, 145.363, 52.5613], [61.0333, 69.4621, 199.695], [13.2044, 143.928, 75.2651], [114.965, 11.9752, 126.441], [42.9649, 146.385, 16.7422], [19.3317, 147.391, 8.5252], [96.0175, 136.792, 13.7001], [75.0489, 15.8462, 86.0371], [133.388, 35.4342, 107.282], [99.0594, 84.8399, 42.7193], [78.7372, 84.8016, 45.4438], [97.4891, 64.5915, 119.992], [17.5065, 129.217, 74.3741], [87.4626, 19.8873, 141.787], [120.782, 123.85, 78.9527], [3.65499, 148.845, 109.868], [105.627, 8.31827, 66.942], [132.821, 79.0867, 55.3622], [144.974, 145.004, 39.6759], [52.2524, 135.224, 22.3862], [40.4073, 147.494, 26.8946], [112.194, 65.6909, 184.821], [31.7585, 17.9317, 166.713], [104.492, 145.374, 35.5374], [100.197, 96.5337, 125.768], [34.3161, 143.62, 114.916], [56.8906, 51.0687, 100.462], [129.32, 148.319, 60.0308], [61.0105, 4.28891, 133.657], [102.906, 114.721, 27.1922], [147.725, 22.6696, 57.0892], [102.713, 84.2056, 84.3396], [141.388, 147.044, 176.595], [131.05, 2.18532, 155.046], [119.506, 55.3924, 148.512], [108.841, 75.2932, 48.7312], [128.806, 18.6162, 67.0497], [106.879, 67.9063, 138.362], [51.6424, 98.3511, 129.901], [126.371, 54.953, 27.8878], [130.575, 147.841, 36.5355]])

xs = data[:,0]
ys = data[:,1]
zs = data[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.scatter(xs, ys, zs)

# plt.show()

for point in data:
  print('ivec3(%d, %d, %d), ' % (point[0], point[1], point[2]), end='')