#include <iostream>
#include <string>
#include <random>
#include <numeric>
#include <functional>
#include <execution>
#include <mpi.h>

using namespace std;
constexpr int MIN_ELEM = -1000, MAX_ELEM = 1000;

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cerr << "You must give 1 argument - size of arrays" << endl;
		return -1;
	}
	const int n = atoi(argv[1]);
	vector<float> a(n), x(n);

	MPI_Init(&argc, &argv);

	int procCnt, curProcRank, curElems;
	MPI_Comm_size(MPI_COMM_WORLD, &procCnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &curProcRank);
	curElems = n / procCnt;
	if (curProcRank < n % procCnt)
		++curElems;
	const string procStr = "Process " + std::to_string(curProcRank) + ": ";
	if (curProcRank == 0)
	{
		cout << "Number of processes: " << procCnt << endl;
		cout << "Number of matrix data elements N: " << n << endl;
	}
	cout << procStr << "processing " << curElems << " elements" << endl;
	
	double startTime, endTime;
	if (curProcRank == 0)
	{
		random_device rd;
		mt19937 engine(rd());
		uniform_real_distribution<float> dist(MIN_ELEM, MAX_ELEM);
		auto genFn = bind(ref(dist), ref(engine));
		generate(begin(a), end(a), genFn);
		generate(begin(x), end(x), genFn);

		startTime = MPI_Wtime();
	}

	vector<float> ap(curElems), xp(curElems);
	MPI_Scatter(&x[0], curElems, MPI_FLOAT, &xp[0], curElems, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(&a[0], curElems, MPI_FLOAT, &ap[0], curElems, MPI_FLOAT, 0, MPI_COMM_WORLD);
	const float sum = inner_product(begin(ap), end(ap), begin(xp), 0.0f);
	float total = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&sum, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (curProcRank == 0)
	{
		endTime = MPI_Wtime();
		cout << endTime - startTime << "s elapsed" << endl;
		cout << "Overall dot product of a and x is: " << total << endl;
		cout << "Dot product using std::transform_reduce: " << transform_reduce(
			execution::par, begin(a), end(a), begin(x), 0.0f) << endl;
	}

	MPI_Finalize();
	return 0;
}