#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>

using namespace std;

const int UpperBound = 10086;

// return the index permutation after sorting
vector<int> sort_indexes_e(vector<double> &v)
{
    vector<int> idx(v.size());
    for (int i = 0; i < v.size(); i++)
        idx[i] = i;
    sort(idx.begin(),idx.end(), [&v](int i1, int i2) {return v[i1] > v[i2]; });
    return idx;
}

vector<vector<int> > HeuristicSearch(int T,int N,const double* c,int* d,int* e,int* l,const double* a_hat){
    vector<vector<int> > X(N, vector<int>(T));
    vector<double> VM_queue(N);

    // setup VM_queue
    for (int i = 0; i < N; i++)
    {
        VM_queue[i] = c[i]/d[i];
        //VM_queue[i] = c[i]*d[i];
    }
    vector<int> sorted_index = sort_indexes_e(VM_queue);
    int counter = 0;

    vector<double> load_list(T);

    for (int i = 0; i < N; i++)
    {
        int j = sorted_index[i];

        int num_delta = min(l[j], T - d[j]) + 1 - e[j];
        vector<double> delta_list(num_delta);
        vector<double> cur_delta_list(d[j]);

        for (int start_t = e[j]; start_t < min(l[j], T - d[j]) + 1; start_t++)
        {
            for (int cur_timestamp = start_t; cur_timestamp < start_t + d[j]; cur_timestamp++)
            {
                double ahead_load = load_list[cur_timestamp] + c[j];
                cur_delta_list[cur_timestamp - start_t] = a_hat[cur_timestamp] - ahead_load;
            }
            double min_delta = *min_element(cur_delta_list.begin(),cur_delta_list.end());
            if (min_delta < 0){
                min_delta = -1;
            }
            delta_list[start_t - e[j]] = min_delta;
        }
        double optimal_delta = *max_element(delta_list.begin(),delta_list.end());
        if (optimal_delta == -1)
        {
            continue;
        }else
        {
            vector<double>::iterator iter;
            iter = max_element(delta_list.begin(),delta_list.end());
            int res = iter - delta_list.begin();
            int opt_timestamp = e[j] + res;
            counter += 1;
            X[j][opt_timestamp] = 1;
            for (int time = opt_timestamp; time < opt_timestamp + d[j]; ++time) {
                load_list[time] += c[j];
            }
        }
    }
    return X;
}

int main(int argc, char * argv[]){
    string file_name = argv[1];
    ifstream infile;
    infile.open(file_name+"_in.txt");

    int T, N;
    infile >> T >> N;
    double* c = new double[N];
    int* d = new int[N];
    int* e = new int[N];
    int* l = new int[N];
    double* a_hat = new double[T];

    for (int i = 0; i < N; ++i) {
        infile >> c[i];
    }
    for (int i = 0; i < N; ++i) {
        infile >> d[i];
    }
    for (int i = 0; i < N; ++i) {
        infile >> e[i];
    }
    for (int i = 0; i < N; ++i) {
        infile >> l[i];
    }
    for (int i = 0; i < T; ++i) {
        infile >> a_hat[i];
    }

    infile.close();

    vector<vector<int> > ans = HeuristicSearch( T, N, c, d, e, l, a_hat);

    ofstream outfile;
    outfile.open(file_name+"_out.txt");

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < T; ++j) {
            outfile << ans[i][j] << " ";
        }
        outfile << endl;
    }
    outfile.close();

    delete[] c;
    delete[] d;
    delete[] e;
    delete[] l;
    delete[] a_hat;

    return 0;
}