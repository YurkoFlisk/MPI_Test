#include <iostream>
#include <string>
#include <mpi.h>

using namespace std;
constexpr int N = 100;

int main(int argc, char* argv[])
{
	unsigned char buffer[N]{};
	int position = 0, curProcRank;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &curProcRank);
	const string procStr = "Process " + std::to_string(curProcRank) + ": ";

	if (curProcRank == 0)
	{
		double x = 43.4, y = 12.1;
		int a[3] = { 123, 84, 67 };
		char message[] = "Hello from process 0";
		constexpr int messageSize = sizeof(message) / sizeof(char);

		MPI_Pack(&x, 1, MPI_DOUBLE, buffer, N, &position, MPI_COMM_WORLD);
		MPI_Pack(&y, 1, MPI_DOUBLE, buffer, N, &position, MPI_COMM_WORLD);
		MPI_Pack(a, 3, MPI_INT, buffer, N, &position, MPI_COMM_WORLD);
		MPI_Pack(&messageSize, 1, MPI_INT, buffer, N, &position, MPI_COMM_WORLD);
		MPI_Pack(message, messageSize, MPI_CHAR, buffer, N, &position, MPI_COMM_WORLD);
	}

	MPI_Bcast(buffer, N, MPI_PACKED, 0, MPI_COMM_WORLD);

	position = 0;
	double x, y;
	int a[3], messageSize;
	MPI_Unpack(buffer, N, &position, &x, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, N, &position, &y, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, N, &position, a, 3, MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buffer, N, &position, &messageSize, 1, MPI_INT, MPI_COMM_WORLD);
	string message(messageSize - 1, '\0');
	MPI_Unpack(buffer, N, &position, message.data(), messageSize, MPI_CHAR, MPI_COMM_WORLD);

	cout << procStr
		<< "\n\tx = " << x << ", y = " << y
		<< "\n\ta = " << a[0] << ' ' << a[1] << ' ' << a[2]
		<< "\n\tmessage = " << message << endl;

	MPI_Finalize();
	return 0;
}