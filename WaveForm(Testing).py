import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# Directory where the extracted clips are stored
extracted_audio_dir = "D:/Elephants Documentation/Output_Audios/ClearingOutput/FinalClips(2 secs)"
output_plot_dir = "D:/Elephants Documentation/Output_Audios/ClearingOutput/FinalWaveForms"

# the output directory for plots exists
os.makedirs(output_plot_dir, exist_ok=True)

# List of specific extracted audio clips
extracted_clips = [
    "clip_1.wav", "clip_2.wav", "clip_3.wav", "clip_4.wav", "clip_5.wav", "clip_6.wav", "clip_7.wav", "clip_8.wav", "clip_9.wav", "clip_10.wav", "clip_11.wav", "clip_12.wav", "clip_13.wav", "clip_14.wav", "clip_15.wav", "clip_16.wav", "clip_17.wav", "clip_18.wav", "clip_19.wav", "clip_20.wav", "clip_21.wav", "clip_22.wav", "clip_23.wav", "clip_24.wav", "clip_25.wav", "clip_26.wav", "clip_27.wav", "clip_28.wav", "clip_29.wav", "clip_30.wav", "clip_31.wav", "clip_32.wav", "clip_33.wav", "clip_34.wav", "clip_35.wav", "clip_36.wav", "clip_37.wav", "clip_38.wav", "clip_39.wav", "clip_40.wav", "clip_41.wav", "clip_42.wav", "clip_43.wav", "clip_44.wav", "clip_45.wav", "clip_46.wav", "clip_47.wav", "clip_48.wav", "clip_49.wav", "clip_50.wav", "clip_51.wav", "clip_52.wav", "clip_53.wav", "clip_54.wav", "clip_55.wav", "clip_56.wav", "clip_57.wav", "clip_58.wav", "clip_59.wav", "clip_60.wav",
    "clip_61.wav", "clip_62.wav", "clip_63.wav", "clip_64.wav", "clip_65.wav", "clip_66.wav", "clip_67.wav", "clip_68.wav", "clip_69.wav", "clip_70.wav", "clip_71.wav", "clip_72.wav", "clip_73.wav", "clip_74.wav", "clip_75.wav", "clip_76.wav", "clip_77.wav", "clip_78.wav", "clip_79.wav", "clip_80.wav","clip_81.wav", "clip_82.wav", "clip_83.wav", "clip_84.wav", "clip_85.wav", "clip_86.wav", "clip_87.wav", "clip_88.wav", "clip_89.wav", "clip_90.wav", "clip_91.wav", "clip_92.wav", "clip_93.wav", "clip_94.wav", "clip_95.wav", "clip_96.wav", "clip_97.wav", "clip_98.wav", "clip_99.wav", "clip_100.wav", "clip_101.wav", "clip_102.wav", "clip_103.wav", "clip_104.wav", "clip_105.wav", "clip_106.wav", "clip_107.wav", "clip_108.wav", "clip_109.wav", "clip_110.wav", "clip_111.wav", "clip_112.wav", "clip_113.wav", "clip_114.wav", "clip_115.wav", "clip_116.wav", "clip_117.wav", "clip_118.wav", "clip_119.wav", "clip_120.wav", "clip_121.wav", "clip_122.wav", "clip_123.wav", "clip_124.wav", "clip_125.wav", "clip_126.wav", "clip_127.wav", "clip_128.wav", "clip_129.wav", "clip_130.wav", "clip_131.wav", "clip_132.wav", "clip_133.wav", "clip_134.wav", "clip_135.wav", "clip_136.wav", "clip_137.wav", "clip_138.wav", "clip_139.wav", "clip_140.wav", "clip_141.wav", "clip_142.wav", "clip_143.wav", "clip_144.wav", "clip_145.wav", "clip_146.wav", "clip_147.wav", "clip_148.wav", "clip_149.wav", "clip_150.wav", "clip_151.wav", "clip_152.wav", "clip_153.wav", "clip_154.wav", "clip_155.wav", "clip_156.wav", "clip_157.wav", "clip_158.wav", "clip_159.wav", "clip_160.wav", "clip_161.wav", "clip_162.wav", "clip_163.wav", "clip_164.wav", "clip_165.wav", "clip_166.wav", "clip_167.wav", "clip_168.wav", "clip_169.wav", "clip_170.wav", "clip_171.wav", "clip_172.wav", "clip_173.wav", "clip_174.wav", "clip_175.wav", "clip_176.wav", "clip_177.wav", "clip_178.wav", "clip_179.wav", "clip_180.wav", "clip_181.wav", "clip_182.wav", "clip_183.wav", "clip_184.wav", "clip_185.wav", "clip_186.wav", "clip_187.wav", "clip_188.wav", "clip_189.wav", "clip_190.wav", "clip_191.wav", "clip_192.wav", "clip_193.wav", "clip_194.wav", "clip_195.wav", "clip_196.wav", "clip_197.wav", "clip_198.wav", "clip_199.wav", "clip_200.wav", "clip_201.wav", "clip_202.wav", "clip_203.wav", "clip_204.wav", "clip_205.wav", "clip_206.wav", "clip_207.wav", "clip_208.wav", "clip_209.wav", "clip_210.wav", "clip_211.wav", "clip_212.wav", "clip_213.wav", "clip_214.wav", "clip_215.wav", "clip_216.wav", "clip_217.wav", "clip_218.wav", "clip_219.wav", "clip_220.wav", "clip_221.wav", "clip_222.wav", "clip_223.wav", "clip_224.wav", "clip_225.wav", "clip_226.wav", "clip_227.wav", "clip_228.wav", "clip_229.wav", "clip_230.wav", "clip_231.wav", "clip_232.wav", "clip_233.wav", "clip_234.wav", "clip_235.wav", "clip_236.wav", "clip_237.wav", "clip_238.wav", "clip_239.wav", "clip_240.wav", "clip_241.wav", "clip_242.wav", "clip_243.wav", "clip_244.wav", "clip_245.wav", "clip_246.wav", "clip_247.wav", "clip_248.wav", "clip_249.wav", "clip_250.wav", "clip_251.wav", "clip_252.wav", "clip_253.wav", "clip_254.wav", "clip_255.wav", "clip_256.wav", "clip_257.wav", "clip_258.wav", "clip_259.wav", "clip_260.wav", "clip_261.wav", "clip_262.wav", "clip_263.wav", "clip_264.wav", "clip_265.wav", "clip_266.wav", "clip_267.wav", "clip_268.wav", "clip_269.wav", "clip_270.wav", "clip_271.wav", "clip_272.wav", "clip_273.wav", "clip_274.wav", "clip_275.wav", "clip_276.wav", "clip_277.wav", "clip_278.wav", "clip_279.wav", "clip_280.wav", "clip_281.wav", "clip_282.wav", "clip_283.wav", "clip_284.wav", "clip_285.wav", "clip_286.wav", "clip_287.wav", "clip_288.wav", "clip_289.wav", "clip_290.wav", "clip_291.wav", "clip_292.wav", "clip_293.wav", "clip_294.wav", "clip_295.wav", "clip_296.wav", "clip_297.wav", "clip_298.wav", "clip_299.wav", "clip_300.wav", "clip_301.wav", "clip_302.wav", "clip_303.wav", "clip_304.wav", "clip_305.wav", "clip_306.wav", "clip_307.wav", "clip_308.wav", "clip_309.wav", "clip_310.wav", "clip_311.wav", "clip_312.wav", "clip_313.wav", "clip_314.wav", "clip_315.wav", "clip_316.wav", "clip_317.wav", "clip_318.wav", "clip_319.wav", "clip_320.wav", "clip_321.wav", "clip_322.wav", "clip_323.wav", "clip_324.wav", "clip_325.wav", "clip_326.wav", "clip_327.wav", "clip_328.wav", "clip_329.wav", "clip_330.wav", "clip_331.wav", "clip_332.wav", "clip_333.wav", "clip_334.wav", "clip_335.wav", "clip_336.wav", "clip_337.wav", "clip_338.wav", "clip_339.wav", "clip_340.wav", "clip_341.wav", "clip_342.wav", "clip_343.wav", "clip_344.wav", "clip_345.wav", "clip_346.wav", "clip_347.wav", "clip_348.wav", "clip_349.wav", "clip_350.wav", "clip_351.wav", "clip_352.wav", "clip_353.wav", "clip_354.wav", "clip_355.wav", "clip_356.wav", "clip_357.wav", "clip_358.wav", "clip_359.wav", "clip_360.wav", "clip_361.wav", "clip_362.wav", "clip_363.wav", "clip_364.wav", "clip_365.wav", "clip_366.wav", "clip_367.wav", "clip_368.wav", "clip_369.wav", "clip_370.wav", "clip_371.wav", "clip_372.wav", "clip_373.wav", "clip_374.wav", "clip_375.wav", "clip_376.wav", "clip_377.wav", "clip_378.wav", "clip_379.wav", "clip_380.wav", "clip_381.wav", "clip_382.wav", "clip_383.wav", "clip_384.wav", "clip_385.wav", "clip_386.wav", "clip_387.wav", "clip_388.wav", "clip_389.wav", "clip_390.wav", "clip_391.wav", "clip_392.wav", "clip_393.wav", "clip_394.wav", "clip_395.wav", "clip_396.wav", "clip_397.wav", "clip_398.wav", "clip_399.wav", "clip_400.wav", "clip_401.wav", "clip_402.wav", "clip_403.wav", "clip_404.wav", "clip_405.wav", "clip_406.wav", "clip_407.wav", "clip_408.wav", "clip_409.wav", "clip_410.wav", "clip_411.wav", "clip_412.wav", "clip_413.wav", "clip_414.wav", "clip_415.wav", "clip_416.wav", "clip_417.wav", "clip_418.wav", "clip_419.wav", "clip_420.wav", "clip_421.wav", "clip_422.wav", "clip_423.wav", "clip_424.wav", "clip_425.wav", "clip_426.wav", "clip_427.wav", "clip_428.wav", "clip_429.wav", "clip_430.wav", "clip_431.wav", "clip_432.wav", "clip_433.wav", "clip_434.wav", "clip_435.wav", "clip_436.wav", "clip_437.wav", "clip_438.wav", "clip_439.wav", "clip_440.wav", "clip_441.wav", "clip_442.wav", "clip_443.wav", "clip_444.wav", "clip_445.wav", "clip_446.wav", "clip_447.wav", "clip_448.wav", "clip_449.wav", "clip_450.wav", "clip_451.wav", "clip_452.wav", "clip_453.wav", "clip_454.wav", "clip_455.wav", "clip_456.wav", "clip_457.wav", "clip_458.wav", "clip_459.wav", "clip_460.wav", "clip_461.wav", "clip_462.wav", "clip_463.wav", "clip_464.wav", "clip_465.wav", "clip_466.wav", "clip_467.wav", "clip_468.wav", "clip_469.wav", "clip_470.wav", "clip_471.wav", "clip_472.wav", "clip_473.wav", "clip_474.wav", "clip_475.wav", "clip_476.wav", "clip_477.wav", "clip_478.wav", "clip_479.wav", "clip_480.wav", "clip_481.wav", "clip_482.wav", "clip_483.wav", "clip_484.wav", "clip_485.wav", "clip_486.wav", "clip_487.wav", "clip_488.wav", "clip_489.wav", "clip_490.wav", "clip_491.wav", "clip_492.wav", "clip_493.wav", "clip_494.wav", "clip_495.wav", "clip_496.wav", "clip_497.wav", "clip_498.wav", "clip_499.wav", "clip_500.wav", "clip_501.wav", "clip_502.wav", "clip_503.wav", "clip_504.wav", "clip_505.wav", "clip_506.wav", "clip_507.wav", "clip_508.wav", "clip_509.wav", "clip_510.wav", "clip_511.wav", "clip_512.wav", "clip_513.wav", "clip_514.wav", "clip_515.wav", "clip_516.wav", "clip_517.wav", "clip_518.wav", "clip_519.wav", "clip_520.wav", "clip_521.wav", "clip_522.wav"
]

# Low-pass filter parameters
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Normalize the audio data
def normalize(data):
    data = data / np.max(np.abs(data), axis=0)
    return data

def plot_waveform(wav_file, output_plot_dir, clip_counter):
    rate, data = wavfile.read(wav_file)
    
    # Apply low-pass filter
    cutoff_frequency = 20.0  # Infrasound cutoff frequency in Hz
    filtered_data = lowpass_filter(data, cutoff_frequency, rate)
    
    # Normalize the data
    normalized_data = normalize(filtered_data)
    
    # Create time array in seconds
    time = np.linspace(0, len(data) / rate, num=len(data))
    
    plt.figure(figsize=(12, 4))
    plt.plot(time, normalized_data)
    plt.title(f'Waveform for {wav_file}')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.grid()

    # Adjust y-axis limits to zoom in
    # plt.ylim(-1, 1)  # Adjust these values as needed

    plt.xlim(0, time[-1])  # Set x-axis limits
    plt.savefig(os.path.join(output_plot_dir, f"waveform_{clip_counter}.png"))
    plt.close()

# Process each specified extracted audio clip
for clip_counter, clip in enumerate(extracted_clips, start=1):
    wav_file_path = os.path.join(extracted_audio_dir, clip)
    plot_waveform(wav_file_path, output_plot_dir, clip_counter)

print("Done: All waveform plots have been saved!")
