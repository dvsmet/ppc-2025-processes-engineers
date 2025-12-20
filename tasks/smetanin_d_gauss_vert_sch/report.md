# Метод Гаусса — ленточная вертикальная схема Сметанин Дмитрий

- Student: Сметанин Дмитрий Владимирович, 3823Б1ПР3
- Technology: SEQ | MPI
- Variant: 16

## 1. Introduction
В этой работе реализованы два варианта решения задачи решения системы линейных уравнений с расширенной матрицей в ленточной форме: последовательный (`SEQ`) и параллельный (`MPI`). Цель — реализовать алгоритм, проверить корректность (функциональные тесты) и провести замеры производительности для разных чисел процессов.

## 2. Problem Statement
Дана система линейных уравнений размера `n` с ленточной матрицей и правой частью (augmented matrix). Требуется найти решение методом Гаусса с выбором опорного элемента по столбцу, при этом учитывать ленточную структуру (ширина полосы `bandwidth`) для ограничения операций.

Вход: `GaussBandInput` (поля `n`, `bandwidth`, `augmented_matrix`).
Выход: вектор неизвестных `OutType` длины `n`.

## 3. Baseline Algorithm (Sequential)
Обычный метод Гаусса (с выбором опорного элемента) над расширенной матрицей, но с ограничением операций по диапазону столбцов и строк, определяемых полосой (`bandwidth`). Алгоритм выполняет:
- поиск опорного элемента в пределах полосы;
- обмен строк (в пределах полосы);
- вычитание строк ниже текущей (с учётом полосы);
- обратную подстановку (back substitution) с учётом полосы.

Сложность в худшем случае близка к O(n * bandwidth). Для узкой полосы это значительно дешевле, чем полная матрица.

## 4. Parallelization Scheme
Параллельная версия реализована как централизованное вычисление на ранге 0 с последующей рассылкой результата всем процессам:
- ранг 0 выполняет всю операцию решения (SolveGaussBand) на входной системе;
- после успешного решения результат (вектор решения) широковещательно отправляется всем процессам (`MPI_Bcast`);
- все ранги получают и помещают решение в `GetOutput()`.

## 5. Implementation Details
tasks\smetanin_d_gauss_vert_sch\
├── common
│ └── include
│ └── common.hpp
├── info.json
├── mpi
│   ├── include
│   │   └── ops_mpi.hpp
│   └── src
│       └── ops_mpi.cpp
├── report.md
├── seq
│   ├── include
│   │   └── ops_seq.hpp
│   └── src
│       └── ops_seq.cpp
├── settings.json
└── tests
    ├── functional
    │   └── functional.cpp
    └── performance
        └── performance.cpp
- `tasks/smetanin_d_gauss_vert_sch/common/include/common.hpp` — общие типы/константы.
- `tasks/smetanin_d_gauss_vert_sch/seq/include/ops_seq.hpp` и `seq/src/ops_seq.cpp` — последовательная реализация.
- `tasks/smetanin_d_gauss_vert_sch/mpi/include/ops_mpi.hpp` и `mpi/src/ops_mpi.cpp` — MPI-обёртка и рассылка результата.
- `tasks/smetanin_d_gauss_vert_sch/tests/functional/main.cpp` — функциональные тесты.
- `tasks/smetanin_d_gauss_vert_sch/tests/performance/main.cpp` — перфоманс тесты (pipeline/task_run).

## 6. Experimental Setup
Hardware/OS: AMD Ryzen 5 5600H, Cores/Threads: 6/12, 16GB RAM, Windows 11 x64.
Toolchain: Microsoft Visual C++, Visual Studio Code 2019/2022, Release, Microsoft MPI 10.1.
Environment: mpiexec -n N, MPI_COMM_WORLD.

## 7. Results and Discussion

Функциональные тесты (запуск `mpiexec -n 4 .\build\bin\ppc_func_tests.exe`):
- Всего выполнено 16 тестов из 2 наборов — все пройдены: все тесты `GaussBandTests` и `SentenceCountTest` прошли успешно.

Перфоманс тесты (четыре запуска: mpiexec -n 1,2,4). В таблицах "Count" — число MPI-процессов при запуске (P).

Pipeline mode (время, скорость):

|      Mode      | P |  Time (s)  |       Speedup (seq/mpi)       | Efficiency (%) |
|----------------|--:|-----------:|------------------------------:|---------------:|
| pipeline (mpi) | 1 | 0.01560458 |             1.01x             | 101.37%        |
| pipeline (mpi) | 2 | 0.01541806 |             1.61x             | 80.48%         |
| pipeline (mpi) | 4 | 0.01551880 |             3.26x             | 81.45%         |

Task run mode (время, скорость):
|      Mode      | P |  Time (s)  |       Speedup (seq/mpi)       | Efficiency (%) |
|----------------|--:|-----------:|------------------------------:|---------------:|
| task_run (mpi) | 1 | 0.01492284 |             1.15x             | 115.09%        |
| task_run (mpi) | 2 | 0.01652152 |             1.52x             | 75.90%         |
| task_run (mpi) | 4 | 0.01621300 |             3.82x             | 95.42%         |

Примечания по таблицам:
- `Speedup` рассчитан как отношение времени последовательной версии (`seq`) к времени `mpi`.
- `Efficiency` = Speedup / P * 100%.

## 8. Conclusions
- Для P=1 оба режима дают близкие времена (speedup ≈1.0–1.15) — значит накладные расходы MPI и логика кода сопоставимы с последовательной реализацией.
- Для P=2 и P=4: speedup от ~1.5× до ~3.8× в зависимости от режима и набора замеров. Эффективность в большинстве случаев остаётся хорошей (≈76–95%), что говорит о том, что на выбранных входах параллелизация даёт выигрыш.
- Самая высокая относительная выгода наблюдается для больших относительных различий между последовательным `seq` и `mpi`.

## 9. References
1. Материалы курса: <https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html>.
2. Microsoft MPI: <https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-reference>.
3. OpenMPI документация: <https://www.open-mpi.org/>.
4. Сысоев А. В. *Лекции по параллельному программированию*.