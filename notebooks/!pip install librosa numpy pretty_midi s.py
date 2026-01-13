!pip install librosa numpy pretty_midi scipy
!pip install demucs
!pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# @title
"""
VOCAL ANALYZER PRO
Анализ вокальных записей: определение типа голоса, извлечение нот, тайминг экстремальных нот
"""

import librosa
import numpy as np
import pretty_midi
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import subprocess
import tempfile
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключаем логи TensorFlow


from google.colab import drive
drive.mount('/content/drive/')
# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
class Config:
    """Настройки анализатора"""
    # Папки Google Drive (для Colab)
    GOOGLE_DRIVE_PATHS = [
        "/content/drive/MyDrive/Вокал/"
    ]

    # ОТДЕЛЕНИЕ ВОКАЛА - ИЗМЕНЕНО
    SEPARATION_METHOD = 'simple'  # 'demucs', 'spleeter', 'simple' или 'none'
  # 'demucs', 'spleeter' или 'none'
    DEMUCS_MODEL = 'htdemucs'  # Модель для demucs
    SEPARATION_OUTPUT_DIR = None  # Если None - будет использоваться временная папка

    SAVE_SEPARATED_VOCAL = True  # Сохранять ли отделённый вокал
    VOCAL_OUTPUT_FORMAT = 'wav'  # 'wav', 'mp3', 'flac'
    OUTPUT_DIR = "/content/drive/MyDrive/Вокал/Анализ/"  # Папка для сохранения
    # Параметры извлечения питча

    GOOGLE_DRIVE_PATHS = [
        "/content/drive/MyDrive/Вокал/"
    ]

    # Параметры извлечения питча
    PITCH_EXTRACTION_METHOD = 'pyin'  # 'pyin' (точнее) или 'piptrack' (быстрее)
    PYIN_FMIN = 65.0  # Минимальная частота для мужских голосов (C2)
    PYIN_FMAX = 1047.0  # Максимальная частота для женских (C6)

    # Фильтрация
    MIN_NOTE_DURATION = 0.05  # Минимальная длительность ноты в секундах
    CONFIDENCE_THRESHOLD = 0.3  # Порог уверенности для pyin

    # Сглаживание
    SMOOTH_WINDOW = 11  # Размер окна медианного фильтра (должно быть нечётным)

    # Для определения типа голоса
    VOICE_TYPES = {
        # Мужские голоса (MIDI-диапазоны)
        'bass': {'range': (36, 55), 'tessitura': (40, 48), 'names': ['Бас', 'Низкий бас', 'Высокий бас']},
        'bass_baritone': {'range': (40, 60), 'tessitura': (44, 52), 'names': ['Бас-баритон', 'Низкий баритон']},
        'baritone': {'range': (45, 65), 'tessitura': (48, 56), 'names': ['Баритон', 'Лирический баритон', 'Драматический баритон']},
        'tenor': {'range': (50, 72), 'tessitura': (52, 62), 'names': ['Тенор', 'Лирический тенор', 'Драматический тенор']},
        'countertenor': {'range': (55, 77), 'tessitura': (58, 68), 'names': ['Контратенор', 'Альтино']},

        # Женские голоса
        'contralto': {'range': (48, 68), 'tessitura': (52, 60), 'names': ['Контральто', 'Низкое меццо']},
        'mezzo': {'range': (55, 75), 'tessitura': (58, 66), 'names': ['Меццо-сопрано', 'Лирическое меццо', 'Драматическое меццо']},
        'soprano': {'range': (60, 84), 'tessitura': (64, 72), 'names': ['Сопрано', 'Лирическое сопрано', 'Драматическое сопрано', 'Колоратурное сопрано']},
    }

# ============================================================================
# УТИЛИТЫ
# ============================================================================
def find_audio_file(filename: str, search_paths: List[str] = None) -> str:
    """
    Ищет аудиофайл в указанных папках и подпапках

    Args:
        filename: Имя файла для поиска
        search_paths: Список путей для поиска (если None, используется Config.GOOGLE_DRIVE_PATHS)

    Returns:
        Полный путь к найденному файлу

    Raises:
        FileNotFoundError: Если файл не найден
    """
    if search_paths is None:
        search_paths = Config.GOOGLE_DRIVE_PATHS

    # Добавляем расширения, если их нет в имени файла
    base_name = Path(filename).stem
    possible_names = [filename]
    for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        if not filename.lower().endswith(ext):
            possible_names.append(base_name + ext)
            possible_names.append(base_name + ext.upper())

    # Ищем файл
    for search_path in search_paths:
        search_path = Path(search_path)
        if search_path.exists():
            for name in possible_names:
                # Ищем в корне
                file_path = search_path / name
                if file_path.exists():
                    return str(file_path)

                # Ищем рекурсивно во всех подпапках
                for file in search_path.rglob(name):
                    if file.exists():
                        return str(file)

    raise FileNotFoundError(f"Файл '{filename}' не найден в папках: {search_paths}")

def separate_vocals(input_path: str, output_dir: str = None, save_permanently: bool = False) -> str:
    """
    Отделяет вокал от аккомпанемента с помощью Demucs

    Args:
        input_path: Путь к исходному аудиофайлу
        output_dir: Папка для сохранения результатов (если None - временная папка)

    Returns:
        Путь к файлу с отделённым вокалом
    """
    print(f"🎚️  Отделение вокала с помощью Demucs...")

    if output_dir is None:
        # Создаём временную папку
        output_dir = tempfile.mkdtemp(prefix="vocal_separation_")

    try:
        from demucs import separate
        import shutil

        input_path_obj = Path(input_path)
        output_dir_obj = Path(output_dir)

        print(f"  Используется модель: {Config.DEMUCS_MODEL}")
        print(f"  Обработка файла: {input_path_obj.name}")

        # Запускаем demucs
        separate.main([
            "--two-stems", "vocals",  # отделяем только вокал
            "-n", Config.DEMUCS_MODEL,
            "-o", str(output_dir_obj),
            str(input_path_obj)
        ])

        # Ищем файл с вокалом
        # Demucs создает папку типа htdemucs/имя_файла/vocals.wav
        model_dir = output_dir_obj / Config.DEMUCS_MODEL
        if not model_dir.exists():
            # Ищем любую папку с моделью
            model_dirs = list(output_dir_obj.glob("*"))
            if model_dirs:
                model_dir = model_dirs[0]

        vocal_path = None
        for potential_dir in [model_dir / input_path_obj.stem,
                             model_dir / input_path_obj.with_suffix('').name]:
            if potential_dir.exists():
                for file in potential_dir.glob("vocals.*"):
                    if file.suffix in ['.wav', '.mp3', '.flac']:
                        vocal_path = file
                        break
                if vocal_path:
                    break

        if not vocal_path:
            # Ищем любой файл с вокалом
            for file in model_dir.rglob("*vocals*.*"):
                if file.suffix in ['.wav', '.mp3', '.flac']:
                    vocal_path = file
                    break

        if not vocal_path or not vocal_path.exists():
            raise FileNotFoundError(f"Не найден файл вокала в {output_dir}")


        if save_permanently and Config.SAVE_SEPARATED_VOCAL:
            # Создаём постоянную папку для сохранения
            permanent_dir = Path(Config.OUTPUT_DIR) if Config.OUTPUT_DIR else Path(input_path).parent / "separated_vocals"
            permanent_dir.mkdir(parents=True, exist_ok=True)

            # Сохраняем вокал в постоянную папку
            permanent_path = permanent_dir / f"{Path(input_path).stem}_vocals.{Config.VOCAL_OUTPUT_FORMAT}"

            # Копируем или конвертируем в нужный формат
            import shutil
            if Config.VOCAL_OUTPUT_FORMAT.lower() == 'wav':
                shutil.copy2(final_vocal_path, permanent_path)
            else:
                # Конвертируем в другой формат
                import soundfile as sf
                y, sr = librosa.load(final_vocal_path, sr=None)
                sf.write(permanent_path, y, sr)

            print(f"💾 Вокал сохранён постоянно: {permanent_path}")
            final_vocal_path = permanent_path

        # Копируем вокал в отдельный файл для удобства
        final_vocal_path = output_dir_obj / f"{input_path_obj.stem}_vocals.wav"
        shutil.copy2(vocal_path, final_vocal_path)

        print(f"✅ Вокал отделён и сохранён: {final_vocal_path}")
        return str(final_vocal_path)

    except ImportError:
        print("❌ Demucs не установлен. Установите: !pip install demucs")
        return input_path
    except Exception as e:
        print(f"❌ Ошибка при отделении вокала: {e}")
        print("ℹ️  Возвращаю исходный файл для анализа")
        return input_path

def separate_vocals_simple(input_path: str, output_dir: str = None) -> str:
    """
    Простое отделение вокала через частотные фильтры
    """
    import soundfile as sf

    print(f"🎚️  Простое отделение вокала (частотный фильтр)...")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="vocal_simple_")

    try:
        # Загружаем аудио
        y, sr = librosa.load(input_path, sr=None, mono=True)

        # Применяем bandpass фильтр (основной диапазон вокала)
        # Мужские голоса: 85-300 Гц, женские: 165-525 Гц, общий: 80-1100 Гц

        # Создаем bandpass фильтр
        from scipy import signal
        nyquist = sr / 2

        # Фильтр для вокала (80-1100 Гц)
        low = 80.0 / nyquist
        high = 1100.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')

        # Применяем фильтр
        y_vocals = signal.filtfilt(b, a, y)

        # Сохраняем результат
        output_path = Path(output_dir) / f"{Path(input_path).stem}_vocals_simple.wav"
        sf.write(output_path, y_vocals, sr)

        print(f"✅ Вокал выделен (частотный фильтр): {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return input_path

def format_time(seconds: float) -> str:
    """Форматирует время в MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

def midi_to_note_name(midi_number: int) -> str:
    """Конвертирует MIDI номер в название ноты"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi_number // 12 - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"

def detect_voice_type(pitches: np.ndarray, durations: np.ndarray = None) -> Dict:
    """
    Определяет тип голоса по статистике нот
    """
    if len(pitches) == 0:
        return {
            'type': 'unknown',
            'subtype': 'Неизвестно',
            'confidence': 0.0,
            'details': 'Недостаточно данных'
        }

    # Рассчитываем медиану (основной критерий)
    if durations is not None and len(durations) == len(pitches):
        weights = durations / durations.sum()
        median_pitch = weighted_quantile(pitches, 0.5, weights)
        mean_pitch = np.average(pitches, weights=weights)
    else:
        median_pitch = np.median(pitches)
        mean_pitch = np.mean(pitches)

    min_pitch = np.min(pitches)
    max_pitch = np.max(pitches)

    # Определяем тип ПО МЕДИАННОЙ НОТЕ (основной критерий)
    median_midi = int(round(median_pitch))

    # Упрощённое определение по медиане
    if median_midi < 40:  # Ниже E2
        main_type, subtype = 'bass', 'Низкий бас'
    elif median_midi < 45:  # Ниже A2
        main_type, subtype = 'bass_baritone', 'Бас-баритон'
    elif median_midi < 50:  # Ниже D3
        main_type, subtype = 'baritone', 'Драматический баритон'
    elif median_midi < 55:  # Ниже G3
        main_type, subtype = 'baritone', 'Лирический баритон'
    elif median_midi < 60:  # Ниже C4
        main_type, subtype = 'tenor', 'Драматический тенор'
    elif median_midi < 65:  # Ниже F4
        main_type, subtype = 'tenor', 'Лирический тенор'
    elif median_midi < 70:  # Ниже A#4
        main_type, subtype = 'countertenor', 'Контратенор'
    elif median_midi < 75:  # Ниже D#5
        main_type, subtype = 'soprano', 'Лирическое сопрано'
    else:
        main_type, subtype = 'soprano', 'Колоратурное сопрано'

    # Вычисляем confidence на основе того, насколько хорошо данные соответствуют типу
    # Для вашего случая C3-G5 с медианой A3 (57) - это будет тенор
    confidence = 0.8  # Базовая уверенность

    # Корректируем confidence на основе диапазона
    voice_info = Config.VOICE_TYPES.get(main_type, {'range': (0, 127)})
    expected_range = voice_info['range']

    if min_pitch >= expected_range[0] and max_pitch <= expected_range[1]:
        confidence = min(1.0, confidence + 0.15)
    else:
        confidence = max(0.3, confidence - 0.2)

    return {
        'type': main_type,
        'subtype': subtype,
        'confidence': confidence,
        'mean_pitch': float(mean_pitch),
        'median_pitch': float(median_pitch),
        'min_pitch': int(min_pitch),
        'max_pitch': int(max_pitch),
        'details': f"Диапазон: {midi_to_note_name(min_pitch)} - {midi_to_note_name(max_pitch)}, "
                  f"Медиана: {midi_to_note_name(median_midi)}",
    }



def weighted_quantile(data, quantile, weights=None):
    """Вычисляет взвешенный квантиль"""
    if weights is None:
        return np.percentile(data, quantile * 100)

    idx = np.argsort(data)
    sorted_data = data[idx]
    sorted_weights = weights[idx]

    cumsum = np.cumsum(sorted_weights)
    cutoff = sorted_weights.sum() * quantile

    return sorted_data[np.searchsorted(cumsum, cutoff)]

# ============================================================================
# ОСНОВНОЙ КЛАСС АНАЛИЗАТОРА
# ============================================================================
class VocalAnalyzer:
    """Анализатор вокальных записей"""

    def __init__(self):
        self.audio = None
        self.sr = None
        self.duration = None
        self.notes = None
        self.timings = None
        self.stats = None
        self.voice_info = None

    def load_audio(self, filepath: str, separate_vocals_flag: bool = True):
        """
        Загружает аудиофайл с возможностью отделения вокала

        Args:
            filepath: Путь к аудиофайлу или имя файла для поиска
            separate_vocals_flag: Отделять ли вокал от аккомпанемента
        """
        print(f"🔍 Поиск файла: {filepath}")

        # Если это не полный путь, ищем файл
        if not Path(filepath).exists():
            filepath = find_audio_file(filepath)

        print(f"📁 Загрузка: {filepath}")

        # ОТДЕЛЕНИЕ ВОКАЛА - ИСПРАВЛЕНО
        original_filepath = filepath
        if separate_vocals_flag:
            try:
                if Config.SEPARATION_METHOD == 'demucs':
                    filepath = separate_vocals(filepath, Config.SEPARATION_OUTPUT_DIR)
                    print(f"🎤 Используется отделённый вокал (Demucs) из: {filepath}")
                elif Config.SEPARATION_METHOD == 'simple':
                    filepath = separate_vocals_simple(filepath, Config.SEPARATION_OUTPUT_DIR)
                    print(f"🎤 Используется отделённый вокал (частотный фильтр) из: {filepath}")
                else:
                    print(f"ℹ️  Метод отделения вокала: {Config.SEPARATION_METHOD}, пропускаю отделение")
            except Exception as e:
                print(f"⚠️  Не удалось отделить вокал: {e}")
                print(f"📁 Используется исходный файл: {original_filepath}")

        if separate_vocals_flag and Config.SEPARATION_METHOD in ['demucs', 'simple']:
            save_vocal_flag = save_vocal if save_vocal is not None else Config.SAVE_SEPARATED_VOCAL
            if save_vocal_flag:
                # Передаём флаг сохранения в функцию отделения
                filepath = separate_vocals(filepath, Config.SEPARATION_OUTPUT_DIR,
                                          save_permanently=True)

        # Загружаем аудио
        self.audio, self.sr = librosa.load(filepath, sr=None, mono=True)
        self.duration = len(self.audio) / self.sr

        print(f"✅ Загружено: {self.duration:.1f} сек, {self.sr} Гц")
        print(f"   Размер аудио: {len(self.audio)} семплов")

        return self


    def extract_pitch(self, method: str = None):
        """
        Извлекает высоту тона из аудио

        Args:
            method: Метод извлечения ('pyin' или 'piptrack')
        """
        if method is None:
            method = Config.PITCH_EXTRACTION_METHOD

        print(f"🎵 Извлечение высоты тона ({method})...")

        if method == 'pyin':
            # Более точный метод
            pitches, voiced_flags, voiced_probs = librosa.pyin(
                self.audio,
                fmin=Config.PYIN_FMIN,
                fmax=Config.PYIN_FMAX,
                sr=self.sr,
                fill_na=0.0
            )

            # Конвертируем в MIDI и фильтруем по уверенности
            times = librosa.times_like(pitches, sr=self.sr)
            valid_indices = (voiced_probs > Config.CONFIDENCE_THRESHOLD) & np.isfinite(pitches)

            self.timings = times[valid_indices]
            pitches_hz = pitches[valid_indices]
            self.confidences = voiced_probs[valid_indices]

            # Конвертируем в MIDI
            pitches_midi = librosa.hz_to_midi(pitches_hz)

        else:  # piptrack
            pitches, magnitudes = librosa.piptrack(y=self.audio, sr=self.sr)

            # Находим доминирующий питч в каждый момент времени
            times = librosa.times_like(pitches, sr=self.sr)
            pitches_midi = []
            self.timings = []
            self.confidences = []

            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch_hz = pitches[index, i]
                magnitude = magnitudes[index, i]

                if pitch_hz > 0 and np.isfinite(pitch_hz):
                    midi_note = librosa.hz_to_midi(pitch_hz)
                    if np.isfinite(midi_note):
                        pitches_midi.append(midi_note)
                        self.timings.append(times[i])
                        self.confidences.append(magnitude / magnitudes.max())

            pitches_midi = np.array(pitches_midi)
            self.timings = np.array(self.timings)
            self.confidences = np.array(self.confidences)

        # Фильтруем невалидные значения (на всякий случай)
        valid_indices = np.isfinite(pitches_midi)
        pitches_midi = pitches_midi[valid_indices]
        self.timings = self.timings[valid_indices]

        if len(pitches_midi) > 0:
            # Сглаживание медианным фильтром
            if Config.SMOOTH_WINDOW > 1 and len(pitches_midi) > Config.SMOOTH_WINDOW:
                from scipy.ndimage import median_filter
                pitches_midi = median_filter(pitches_midi, size=Config.SMOOTH_WINDOW)

        self.pitches_midi = pitches_midi

        print(f"✅ Извлечено {len(pitches_midi)} отсчётов высоты тона")

        return self


    def extract_notes(self):
        """
        Группирует непрерывные питчи в ноты
        """
        if self.pitches_midi is None:
            raise ValueError("Сначала выполните extract_pitch()")

        print("🎼 Группировка в ноты...")

        notes = []
        current_note = None
        note_start = 0
        note_pitch = 0

        for i, (time, pitch) in enumerate(zip(self.timings, self.pitches_midi)):
            # Проверяем, что pitch валидный (не nan и не inf)
            if not np.isfinite(pitch):
                continue  # Пропускаем невалидные значения

            if current_note is None:
                current_note = pitch
                note_start = time
                note_pitch = pitch
            elif abs(pitch - current_note) > 2.0:  # Значительное изменение высоты
                # Завершаем предыдущую ноту
                note_end = time
                duration = note_end - note_start

                if duration >= Config.MIN_NOTE_DURATION:
                    # Убеждаемся, что current_note и note_pitch валидны
                    if np.isfinite(current_note) and np.isfinite(note_pitch):
                        if i == 0:
                            avg_pitch = note_pitch
                        else:
                            # Взвешенное среднее с проверкой на валидность
                            avg_pitch = current_note * 0.3 + note_pitch * 0.7

                        # Проверяем, что результат валиден
                        if np.isfinite(avg_pitch):
                            pitch_int = int(round(avg_pitch))
                            # Дополнительная проверка на разумный диапазон MIDI
                            if 0 <= pitch_int <= 127:
                                notes.append({
                                    'start': note_start,
                                    'end': note_end,
                                    'duration': duration,
                                    'pitch': pitch_int,
                                    'pitch_name': midi_to_note_name(pitch_int)
                                })

                # Начинаем новую ноту
                current_note = pitch
                note_start = time
                note_pitch = pitch
            else:
                # Продолжаем текущую ноту
                if np.isfinite(pitch):
                    current_note = (current_note * 0.7 + pitch * 0.3)  # Экспоненциальное сглаживание

        # Добавляем последнюю ноту
        if current_note is not None and len(self.timings) > 0:
            note_end = self.timings[-1]
            duration = note_end - note_start

            if duration >= Config.MIN_NOTE_DURATION and np.isfinite(note_pitch):
                pitch_int = int(round(note_pitch))
                if 0 <= pitch_int <= 127:
                    notes.append({
                        'start': note_start,
                        'end': note_end,
                        'duration': duration,
                        'pitch': pitch_int,
                        'pitch_name': midi_to_note_name(pitch_int)
                    })

        self.notes = notes
        print(f"✅ Выделено {len(notes)} нот")

        return self

    def analyze(self):
        """Выполняет полный анализ"""
        if self.notes is None:
            self.extract_pitch().extract_notes()

        print("📊 Анализ статистики...")

        # Извлекаем данные
        pitches = np.array([note['pitch'] for note in self.notes])
        durations = np.array([note['duration'] for note in self.notes])
        starts = np.array([note['start'] for note in self.notes])

        # Находим экстремальные ноты с таймингами
        min_pitch = np.min(pitches)
        max_pitch = np.max(pitches)
        median_pitch = int(round(np.median(pitches)))
        mean_pitch = int(round(np.mean(pitches)))

        # Находим ВСЕ вхождения нот
        min_indices = np.where(pitches == min_pitch)[0]
        max_indices = np.where(pitches == max_pitch)[0]
        median_indices = np.where(pitches == median_pitch)[0]

        # Получаем тайминги
        min_times = [format_time(starts[idx]) for idx in min_indices]
        max_times = [format_time(starts[idx]) for idx in max_indices]
        median_times = [format_time(starts[idx]) for idx in median_indices[:3]]  # первые 3

        # Основная статистика
        stats = {
            'count': len(pitches),
            'min': int(min_pitch),
            'min_note': midi_to_note_name(min_pitch),
            'min_times': min_times,
            'max': int(max_pitch),
            'max_note': midi_to_note_name(max_pitch),
            'max_times': max_times,
            'median': int(median_pitch),
            'median_note': midi_to_note_name(median_pitch),
            'median_times': median_times,
            'mean': float(np.mean(pitches)),
            'median_val': float(np.median(pitches)),
            'std': float(np.std(pitches)),
            'range': int(max_pitch - min_pitch),
            'duration': self.duration,
            'note_density': len(pitches) / self.duration if self.duration > 0 else 0,
        }

        # Взвешенная статистика (по длительности нот)
        if len(durations) > 0:
            weights = durations / durations.sum()
            stats['weighted_mean'] = float(np.average(pitches, weights=weights))
            stats['weighted_median'] = float(weighted_quantile(pitches, 0.5, weights))
        else:
            stats['weighted_mean'] = stats['mean']
            stats['weighted_median'] = stats['median']

        # Распределение по октавам
        octaves = (pitches // 12).astype(int)
        unique_octaves, octave_counts = np.unique(octaves, return_counts=True)
        stats['octave_distribution'] = dict(zip(unique_octaves, octave_counts))

        # Гистограмма нот
        unique_pitches, pitch_counts = np.unique(pitches, return_counts=True)
        stats['pitch_histogram'] = dict(zip(unique_pitches, pitch_counts))

        self.stats = stats

        # Определяем тип голоса
        self.voice_info = detect_voice_type(pitches, durations)

        print("✅ Анализ завершён")

        return self

    def create_midi(self, output_path: str = "vocal_analysis.mid"):
        """Создает MIDI файл из извлеченных нот"""
        if self.notes is None:
            raise ValueError("Сначала выполните extract_notes()")

        print(f"💾 Создание MIDI: {output_path}")

        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

        for note in self.notes:
            midi_note = pretty_midi.Note(
                velocity=80,
                pitch=note['pitch'],
                start=note['start'],
                end=note['end']
            )
            piano.notes.append(midi_note)

        midi.instruments.append(piano)
        midi.write(output_path)

        print(f"✅ MIDI сохранён: {output_path}")

        return output_path

    def print_report(self):
        """Выводит подробный отчёт"""
        if self.stats is None or self.voice_info is None:
            self.analyze()

        print("\n" + "="*70)
        print("🎤 ПОДРОБНЫЙ АНАЛИЗ ВОКАЛА")
        print("="*70)

        # Основная информация
        print(f"\n📊 ОСНОВНАЯ СТАТИСТИКА:")
        print(f"   Длительность: {self.duration:.1f} сек")
        print(f"   Всего нот: {self.stats['count']}")
        print(f"   Плотность нот: {self.stats['note_density']:.1f} нот/сек")

        # Экстремальные ноты с таймингами
        print(f"\n🎵 ЭКСТРЕМАЛЬНЫЕ НОТЫ:")
        min_times_str = ", ".join(self.stats['min_times'][:5])  # Показываем первые 5
        if len(self.stats['min_times']) > 5:
            min_times_str += f" (и ещё {len(self.stats['min_times']) - 5})"

        max_times_str = ", ".join(self.stats['max_times'][:5])
        if len(self.stats['max_times']) > 5:
            max_times_str += f" (и ещё {len(self.stats['max_times']) - 5})"

        print(f"   🎵 Самая низкая: {self.stats['min_note']} (MIDI: {self.stats['min']})")
        print(f"     Время: {min_times_str}")

        print(f"   🎵 Самая высокая: {self.stats['max_note']} (MIDI: {self.stats['max']})")
        print(f"     Время: {max_times_str}")

        print(f"   🎵 Диапазон: {self.stats['range']} полутонов")

        # Статистика высоты
        print(f"\n📈 СТАТИСТИКА ВЫСОТЫ:")
        print(f"   Средняя нота: {midi_to_note_name(int(round(self.stats['mean'])))} "
              f"(MIDI: {self.stats['mean']:.1f})")
        print(f"   Медианная нота: {midi_to_note_name(int(round(self.stats['median'])))} "
              f"(MIDI: {self.stats['median']:.1f})")
        print(f"   Взвешенное среднее: {midi_to_note_name(int(round(self.stats['weighted_mean'])))} "
              f"(MIDI: {self.stats['weighted_mean']:.1f})")
        print(f"   Стандартное отклонение: {self.stats['std']:.1f} полутонов")

        # Тип голоса
        voice = self.voice_info
        print(f"\n🎤 ТИП ГОЛОСА:")
        confidence_stars = "★" * int(voice['confidence'] * 5) + "☆" * (5 - int(voice['confidence'] * 5))
        print(f"   {voice['subtype']} ({confidence_stars} {voice['confidence']*100:.0f}%)")
        print(f"   {voice['details']}")

        # Диапазон и тесситура
        mean_midi = self.stats['weighted_mean']
        if mean_midi < 45:
            tessitura = "Очень низкая (бас)"
        elif mean_midi < 52:
            tessitura = "Низкая (баритон)"
        elif mean_midi < 60:
            tessitura = "Средняя (тенор/меццо)"
        elif mean_midi < 68:
            tessitura = "Высокая (сопрано)"
        else:
            tessitura = "Очень высокая (колоратура)"

        print(f"   Тесситура: {tessitura}")

        # Распределение по октавам
        print(f"\n🎼 РАСПРЕДЕЛЕНИЕ ПО ОКТАВАМ:")
        for octave, count in sorted(self.stats['octave_distribution'].items()):
            percentage = (count / self.stats['count']) * 100
            bar = "█" * int(percentage / 5)
            print(f"   Октава {octave}: {bar} {count} нот ({percentage:.1f}%)")

        # Самые частые ноты
        if 'pitch_histogram' in self.stats:
            print(f"\n🎶 САМЫЕ ЧАСТЫЕ НОТЫ:")
            sorted_pitches = sorted(self.stats['pitch_histogram'].items(),
                                  key=lambda x: x[1], reverse=True)[:5]
            for pitch, count in sorted_pitches:
                percentage = (count / self.stats['count']) * 100
                print(f"   {midi_to_note_name(pitch)}: {count} раз ({percentage:.1f}%)")

        print("="*70)
        print("✅ Анализ завершён. Для сохранения MIDI используйте .create_midi()")

        return self
    def save_vocal_track(self, output_path: str = None):
        """
        Сохраняет отделённый вокал в файл

        Args:
            output_path: Путь для сохранения (если None - генерируется автоматически)

        Returns:
            Путь к сохранённому файлу
        """
        if self.audio is None:
            raise ValueError("Сначала загрузите аудио")

        import soundfile as sf

        if output_path is None:
            # Автогенерация имени файла
            if Config.OUTPUT_DIR:
                output_dir = Path(Config.OUTPUT_DIR)
            else:
                output_dir = Path.cwd() / "vocal_output"

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"vocal_track_{int(time.time())}.{Config.VOCAL_OUTPUT_FORMAT}"

        # Сохраняем аудио
        sf.write(output_path, self.audio, self.sr)
        print(f"💾 Вокальная дорожка сохранена: {output_path}")

        return str(output_path)

# ============================================================================
# ИНТЕРФЕЙС ДЛЯ GOOGLE COLAB
# ============================================================================
def analyze_vocal(filename: str, mount_drive: bool = True, separate_vocals: bool = True):
    """
    Основная функция для использования в Google Colab

    Args:
        filename: Имя аудиофайла (будет найден в Google Drive)
        mount_drive: Подключать ли Google Drive (True для Colab)
        separate_vocals: Отделять ли вокал от аккомпанемента
    """
    drive.mount('/content/drive')
    print("✅ Google Drive подключен")

    # Создаем и запускаем анализатор
    analyzer = VocalAnalyzer()

    try:
        # ИЗМЕНЕНИЕ: передаём separate_vocals параметр
        analyzer.load_audio(filename, separate_vocals_flag=separate_vocals)  # ИЗМЕНЕНО
        analyzer.extract_pitch()
        analyzer.extract_notes()
        analyzer.analyze()
        analyzer.print_report()

        # Автоматически создаем MIDI
        midi_filename = Path(filename).stem + "_analysis.mid"
        analyzer.create_midi(midi_filename)

        return analyzer

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\n💡 Советы по устранению проблем:")
        print("1. Убедитесь, что файл существует в Google Drive")
        print("2. Проверьте, что это монофоническая вокальная запись")
        print("3. Попробуйте уменьшить CONFIDENCE_THRESHOLD в Config")
        print("4. Увеличьте PYIN_FMIN для мужских голосов")
        print("5. Если отделение вокала не работает, используйте:")
        print("   analyze_vocal('файл.mp3', separate_vocals=False)")
        raise




analyze_vocal("ALizee Jacotey - Coeur Deja Pris-vocals.mp3", separate_vocals=False)