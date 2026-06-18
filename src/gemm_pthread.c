/*
The mathematical logic to slice the output matrix horizontally by row. 
Prevents false sharing by ensuring no two threads write to the same cache line.

size_t base_rows = logical_size / num_threads;
size_t remainder = logical_size % num_threads;
size_t current_row = 0;

for (size_t t = 0; t < num_threads; t++) {
    args[t].start_row = current_row;
    
    size_t thread_rows = base_rows + (t < remainder ? 1 : 0);
    args[t].end_row = current_row + thread_rows;
    
    args[t].logical_size = logical_size;
    args[t].stride       = stride;
    args[t].A            = A->data;
    args[t].B            = B->data;
    args[t].C            = C->data;

    current_row = args[t].end_row;
}

*/