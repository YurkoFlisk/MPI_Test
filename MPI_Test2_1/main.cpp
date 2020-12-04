#include <iostream>
#include <string>
#include <random>
#include <numeric>
#include <functional>
#include <execution>
#include <mpi.h>

using namespace std;
constexpr int N = 60, MIN_ELEM = -1000, MAX_ELEM = 1000;
int a[N], x[N], ap[N], xp[N];

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int procCnt, curProcRank, curElems;
	MPI_Comm_size(MPI_COMM_WORLD, &procCnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &curProcRank);
	curElems = N / procCnt;
	if (curProcRank < N % procCnt)
		++curElems;
	const string procStr = "Process " + std::to_string(curProcRank) + ": ";
	if (curProcRank == 0)
	{
		cout << "Number of processes: " << procCnt << endl;
		cout << "Number of matrix data elements N: " << N << endl;
	}
	cout << procStr << "processing " << curElems << " elements" << endl;

	double startTime, endTime;
	if (curProcRank == 0)
	{
		random_device rd;
		mt19937 engine(rd());
		uniform_int_distribution<int> dist(MIN_ELEM, MAX_ELEM);
		auto genFn = bind(ref(dist), ref(engine));
		generate(begin(a), end(a), genFn);
		generate(begin(x), end(x), genFn);

		cout << "a = ";
		copy(begin(a), end(a), ostream_iterator<int>(cout, " "));
		cout << "\nx = ";
		copy(begin(x), end(x), ostream_iterator<int>(cout, " "));
		cout << endl;

		startTime = MPI_Wtime();
	}

	MPI_Scatter(x, curElems, MPI_INT, xp, curElems, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(a, curElems, MPI_INT, ap, curElems, MPI_INT, 0, MPI_COMM_WORLD);
	const auto apEnd = ap + curElems, xpEnd = xp + curElems;
	const int sum = inner_product(ap, apEnd, xp, 0);

	cout << procStr << "\n\tap = ";
	copy(ap, apEnd, ostream_iterator<int>(cout, " "));
	cout << "\n\txp = ";
	copy(xp, xpEnd, ostream_iterator<int>(cout, " "));
	cout << endl;

	int total = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&sum, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (curProcRank == 0)
	{
		endTime = MPI_Wtime();
		cout << endTime - startTime << "s elapsed" << endl;
		cout << "Overall dot product of a and x is: " << total << endl;
		cout << "Dot product using std::transform_reduce: " << transform_reduce(
			execution::par, begin(a), end(a), begin(x), 0) << endl;
	}

	MPI_Finalize();
	return 0;
}