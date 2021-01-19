#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

const int UpperBound = 10086;

// return the index permutation after sorting
vector<int> sort_indexes_e(vector<double> &v)
{
    vector<int> idx(v.size());
    for (size_t i = 0; i < v.size(); i++)
        idx[i] = static_cast<int>(i);
    sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] > v[i2]; });
    return idx;
}

vector<vector<int>> HeuristicSearch(
    int totalTimeStamps,
    int totalJobNums,
    vector<double> &core,
    vector<int> &duration,
    vector<int> &earliestStartTime,
    vector<int> &latestStartTime,
    vector<double> &coreLimit)
{
    vector<vector<int>> X(totalJobNums, vector<int>(totalTimeStamps));
    vector<double> VM_queue(totalJobNums);

    // setup VM_queue
    for (int i = 0; i < totalJobNums; i++)
    {
        VM_queue[i] = core[i] / duration[i];
    }
    vector<int> sorted_index = sort_indexes_e(VM_queue);
    int counter = 0;

    vector<double> load_list(totalTimeStamps);
    vector<double> delta_list(totalTimeStamps);
    vector<double> cur_delta_list(totalTimeStamps);

    for (int i = 0; i < totalJobNums; i++)
    {
        int j = sorted_index[i];
        int curEaeliest = earliestStartTime[j];
        int curLatestAfter = min(latestStartTime[j], totalTimeStamps - duration[j]) + 1;

        for (int cur_time = earliestStartTime[j]; cur_time < min(latestStartTime[j] + duration[j], totalTimeStamps); cur_time++)
        {
            double ahead_load = load_list[cur_time] + core[j];
            cur_delta_list[cur_time] = coreLimit[cur_time] - ahead_load;
        }

        for (int start_t = curEaeliest; start_t < curLatestAfter; start_t++)
        {
            double min_delta = *min_element(cur_delta_list.begin() + start_t, cur_delta_list.begin() + min(start_t + duration[j], totalTimeStamps));
            delta_list[start_t] = min_delta;
        }

        auto optimal_delta = max_element(delta_list.begin() + curEaeliest, delta_list.begin() + curLatestAfter);
        if (*optimal_delta < 0)
        {
            continue;
        }

        // Allocate VM
        int opt_timestamp = optimal_delta - delta_list.begin();
        counter += 1;
        X[j][opt_timestamp] = 1;
        for (auto load_iter = load_list.begin() + opt_timestamp; load_iter < load_list.begin() + opt_timestamp + duration[j]; ++load_iter)
        {
            *load_iter += core[j];
        }
    }
    return X;
}

PYBIND11_MODULE(heuristic, m)
{
    m.doc() = "Heuristic search based optimization";
    m.def("search", &HeuristicSearch,
          "Heuristic search",
          py::arg("T"),
          py::arg("N"),
          py::arg("c"),
          py::arg("d"),
          py::arg("e"),
          py::arg("l"),
          py::arg("a_hat"));
}
