#include <iostream>
#include <string>
#include <mpi.h>

using namespace std;
constexpr int K = 8;

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	const double beginTime = MPI_Wtime();
	int procCnt, curProcRank;
	MPI_Comm_size(MPI_COMM_WORLD, &procCnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &curProcRank);
	const std::string procStr = "Process " + std::to_string(curProcRank) + ": ";

	cout << procStr << "Number of processes - " << procCnt << endl;
	cout << procStr << "Timer resolution: " << MPI_Wtick() << "s" << endl;
	if (curProcRank % K == 0)
		cout << procStr << "Hello world from each 8-th process" << endl;

	const double endTime = MPI_Wtime();
	cout << procStr << endTime - beginTime << "s elapsed" << endl;

	MPI_Finalize();
	return 0;
}