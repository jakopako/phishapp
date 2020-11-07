from phishapp.detector.logodetector import LogoDetector
import time
import os
import statistics

l_detect = LogoDetector()
start_load_logos = time.time()
l_detect.load_logos('../files/logos')
stop_load_logos = time.time()
print("Took {}s to load {} logos. {}s per logo.".format(
    stop_load_logos-start_load_logos,
    len(l_detect.logo_kps_desc),
    (stop_load_logos-start_load_logos)/len(l_detect.logo_kps_desc)))

screenshot_path = '../../screenshots'
detect_times = []
for filename in os.listdir(screenshot_path):
    if filename.endswith('.png'):
        image = l_detect.open_image_from_path(os.path.join(screenshot_path, filename))
        start_detect = time.time()
        ld = l_detect.find_logos(image)
        stop_detect = time.time()
        detect_times.append(stop_detect-start_detect)
        print('Took {}s to detect logo {} in image {}'.format(stop_detect-start_detect, ld, filename))
print('Took {}s on average.'.format(statistics.mean(detect_times)))