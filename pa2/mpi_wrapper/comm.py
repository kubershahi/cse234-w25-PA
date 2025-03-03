from mpi4py import MPI
import numpy as np


class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )


        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)


        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.

        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.

        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """

        rank = self.comm.Get_rank()
        size = self.comm.Get_size()


        if rank == 0:
            buffer = np.zeros_like(src_array)
        else:
            buffer = None


        if rank == 0:
            np.copyto(buffer, src_array)
            for i in range(1, size):
                recv_buf = np.empty_like(src_array)
                self.comm.Recv(recv_buf, source=i)

                if op == MPI.SUM:
                    buffer += recv_buf  # Apply sum
                elif op == MPI.MIN:
                    buffer = np.minimum(buffer, recv_buf)
                else:
                    raise ValueError("Unsupported operation for myAllreduce")


            self.total_bytes_transferred += src_array.itemsize * src_array.size * (size - 1)

        else:

            self.comm.Send(src_array, dest=0)


            self.total_bytes_transferred += src_array.itemsize * src_array.size


        self.comm.Bcast(buffer if rank == 0 else dest_array, root=0)


        self.total_bytes_transferred += src_array.itemsize * src_array.size * (size - 1)

        if rank == 0:
            np.copyto(dest_array, buffer)

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.

        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.

        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.

        The total data transferred is updated for each pairwise exchange.
        """

        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        assert src_array.size % size == 0, "src_array size must be divisible by the number of processes"
        assert dest_array.size % size == 0, "dest_array size must be divisible by the number of processes"

        segment_size = src_array.size // size
        segment_dtype = src_array.dtype

        for i in range(size):
            send_offset = i * segment_size  # Send correct portion
            recv_offset = i * segment_size  # Receive correct portion

            send_buf = np.copy(src_array[send_offset:send_offset + segment_size])
            recv_buf = np.empty(segment_size, dtype=segment_dtype)

            if i == rank:

                np.copyto(dest_array[recv_offset:recv_offset + segment_size], send_buf)
            else:

                self.comm.Sendrecv(send_buf, dest=i, recvbuf=recv_buf, source=i)
                np.copyto(dest_array[recv_offset:recv_offset + segment_size], recv_buf)

            self.total_bytes_transferred += send_buf.itemsize * send_buf.size
            self.total_bytes_transferred += recv_buf.itemsize * recv_buf.size


