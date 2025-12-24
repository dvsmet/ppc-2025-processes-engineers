# Умножение разреженных матриц. Элементы типа double. Формат хранения матрицы – строковый (CRS). Сметанин Дмитрий

- Студент: Сметанин Дмитрий Владимирович, 3823Б1ПР3
- Технологии: SEQ | MPI
- Вариант: 4

## 1. Введение
В работе реализованы последовательная и параллельная (MPI) версии умножения разреженных матриц в формате CRS с элементами типа double. Параллельная версия демонстрирует реальное ускорение за счёт распределения строк матрицы между процессами.

## 2. Постановка задачи
Цель: Реализовать умножение двух разреженных матриц A и B в формате CRS, получить результат C = A × B также в CRS. Поддержка неквадратных матриц (A: m × k, B: k × n).
Формат CRS:

rows, cols — размеры матрицы
nnz — количество ненулевых элементов
values — ненулевые значения (по строкам)
col_indices — индексы столбцов соответствующих значений
row_ptr — указатели на начало строк в values/col_indices (размер rows + 1)

Вход: std::tuple<CRSMatrix, CRSMatrix> (A, B)
Выход: CRSMatrix (C = A × B)

## 3. Базовый алгоритм (последовательный)
Последовательная реализация выполняет умножение построчно:

Для каждой строки i матрицы A:
Создаётся unordered_map<int, double> для накопления результатов по столбцам.
Перебираются ненулевые элементы A[i][j], для каждого — ненулевые элементы строки j матрицы B.
Накапливается вклад: acc[k] += A[i][j] * B[j][k].

После обработки строки — запись ненулевых элементов из acc в результат C.
Обновление row_ptr[i + 1].

Ограничения алгоритма: Метод не гарантирует сохранения точности при очень малых значениях. Для матриц с высокой плотностью (>50%) эффективность снижается. 
Требует достаточной памяти для accumulator (в худшем случае O(n) на строку).

## 4. Схема распараллеливания
Распределение данных:

Матрица B полностью дублируется на всех процессах (MPI_Bcast).
Строки матрицы A распределяются блоками между процессами (Scatterv для элементов и отдельно для row_ptr).
Каждый процесс вычисляет вклад своих строк A в результат C.

Псевдокод MPI-версии:

textif rank == 0:
    load A, B
broadcast B to all
scatter rows of A to processes
local_rows = my block of rows
local_C = multiply_local_rows(local_A, B_local)
if rank == 0:
    C = local_C
    for p in 1..size-1:
        receive nnz, row_ptr, values, col_indices from p
        insert into C with offset correction
else:
    send local_C parts to root

## 5. Детали реализации
tasks/smetanin_d_multiplication_of_sparse_matrices_storage_format_crs/
├── common/
│   └── include/
│       └── common.hpp          # CRSMatrix (rows, cols, nnz), MultiplyCRS
├── data/
│   └── matrix_generators.hpp   # GenerateRandomCRS
├── seq/
│   ├── include/ops_seq.hpp
│   └── src/ops_seq.cpp         # SEQ версия
├── mpi/
│   ├── include/ops_mpi.hpp
│   └── src/ops_mpi.cpp         # MPI версия
├── tests/
│   ├── functional/main.cpp     # 80 тестов seq-тестов
│   └── performance/main.cpp    # perf-тесты
└── report.md

## 6. Экспериментальная установка
Hardware/OS: AMD Ryzen 5 5600H, Cores/Threads: 6/12, 16GB RAM, Windows 11 x64.
Toolchain: Microsoft Visual C++, Visual Studio Code 2019/2022, Release, Microsoft MPI 10.1.
Environment: mpiexec -n N, MPI_COMM_WORLD.

## 7. Результаты и обсуждение

Функциональные тесты:

Выполнено 80 тестов. Все тесты прошли успешно. Результаты SEQ и MPI версий идентичны.

Перфоманс тесты (шесть запусков: mpiexec -n 1,2,4,7,8). Замеры на больших матрицах:

Pipeline mode (время (примерно), скорость):

|      Mode      | P |  Time (s)  |       Speedup (seq/mpi)       | Efficiency (%) |
|----------------|--:|-----------:|------------------------------:|---------------:|
| pipeline (mpi) | 1 | 1.05621680 |             1.10x             | 110%           |
| pipeline (mpi) | 2 | 0.61853504 |             1.89x             | 94.5%          |
| pipeline (mpi) | 4 | 0.40486800 |             2.90x             | 72.5%          |
| pipeline (mpi) | 7 | 0.32318806 |             3.59x             | 51.3%          |
| pipeline (mpi) | 8 | 0.31273568 |             3.74x             | 46.8%          |

Task run mode (время (примерно), скорость):
Mode,P,Time (s),Speedup (seq/mpi),Efficiency (%)
task_run (mpi),1,~1.06,~1.09x,109%
task_run (mpi),2,~0.60,~1.93x,96.5%
task_run (mpi),4,~0.34,~3.41x,85.3%
task_run (mpi),7,~0.32,~3.63x,51.9%
task_run (mpi),8,~0.30,~3.87x,48.4%

|      Mode      | P |  Time (s)  |       Speedup (seq/mpi)       | Efficiency (%) |
|----------------|--:|-----------:|------------------------------:|---------------:|
| task_run (mpi) | 1 | 1.06586098 |             1.09x             | 109%           |
| task_run (mpi) | 2 | 0.60823440 |             1.93x             | 96.5%          |
| task_run (mpi) | 4 | 0.34458866 |             3.41x             | 85.3%          |
| task_run (mpi) | 7 | 0.32306084 |             3.63x             | 51.9%          |
| task_run (mpi) | 8 | 0.30266608 |             3.87x             | 48.4%          |

Примечания по таблицам:
- `Speedup` рассчитан как отношение времени последовательной версии (`seq`) к времени `mpi`.
- `Efficiency` = Speedup / P * 100%.

## 8. Заключение
- Для P=1 оба режима дают близкие времена (speedup ≈1.09–1.10) — значит накладные расходы MPI и логика кода сопоставимы с последовательной реализацией, с минимальным оверхедом.
- Для P=2,4,7,8: speedup от ~1.9× до ~3.9× в зависимости от режима и набора замеров. Эффективность варьируется от ~48% до ~96%, что говорит о том, что на выбранных входах параллелизация даёт выигрыш, но с ростом P эффективность снижается из-за накладных расходов MPI на коммуникации (broadcast B и send/recv).
- Самая высокая относительная выгода наблюдается при 4 процессах (~3.4x).

## 9. Ссылки
1. Материалы курса: <https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html>.
2. Microsoft MPI: <https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-reference>.
3. OpenMPI документация: <https://www.open-mpi.org/>.
4. Сысоев А. В. *Лекции по параллельному программированию*.
