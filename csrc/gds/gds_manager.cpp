#include "gds_manager.h"
#include "gds_layout_transform.cuh"

#ifdef ENABLE_GDS
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cufile.h>
#endif

#include <vector>
#include <map>

GDSManager::GDSManager(std::map<int, std::vector<std::string>>& ssd_files, 
                       int num_devices, 
                       int round_robin) 
    : is_ready_(false), num_devices_(num_devices), 
      round_robin_(round_robin)
#ifdef ENABLE_GDS
    , driver_initialized_(false), next_batch_id_(1), stop_workers_(false), num_worker_threads_(0)
#endif
{
    if (!initialize_driver()) {
        return;
    }
    
#ifdef ENABLE_GDS
    // Create shared CUDA stream
    cudaError_t cuda_status = cudaStreamCreate(&shared_stream_);
    if (cuda_status != cudaSuccess) {
        set_error("Failed to create shared CUDA stream");
        return;
    }
    
    initialize_worker_threads();
#endif
    
    this->num_files_per_device_ = ssd_files[0].size();
    
    file_paths_.resize(num_devices);
    for (const auto& kv : ssd_files) {
        int device_id = kv.first;
        file_paths_[device_id] = kv.second;
        
        for (const auto& filename : kv.second) {
            add_file(filename.c_str());
        }
    }
 
    is_ready_ = true;
    set_error("GDS Manager initialized successfully");
}

GDSManager::~GDSManager() {
#ifdef ENABLE_GDS
    shutdown_worker_threads();
#endif
    cleanup();
}

bool GDSManager::is_ready() const {
    return is_ready_;
}

const std::string& GDSManager::get_last_error() const {
    return last_error_;
}

int GDSManager::get_num_devices() const {
    return num_devices_;
}

int GDSManager::get_num_files_per_device() const {
    return num_files_per_device_;
}

int GDSManager::get_round_robin() const {
    return round_robin_;
}

void GDSManager::set_error(const std::string& error) {
    last_error_ = error;
}

bool GDSManager::initialize_driver() {
#ifdef ENABLE_GDS
    if (driver_initialized_) {
        return true;
    }
    
    CUfileError_t status = cuFileDriverOpen();
    if (status.err == 0) {
        driver_initialized_ = true;
        return true;
    } else {
        set_error("Failed to initialize cuFile driver");
        return false;
    }
#else
    set_error("GDS support not compiled in (ENABLE_GDS not defined)");
    return false;
#endif
}

bool GDSManager::add_file(const char* filename) {
    if (!filename) {
        set_error("Invalid filename");
        return false;
    }
    
    return open_file_internal(filename);
}

bool GDSManager::remove_file(const char* filename) {
    if (!filename) {
        set_error("Invalid filename");
        return false;
    }
    
    close_file_internal(filename);
    return true;
}

bool GDSManager::open_file_internal(const char* filename) {
#ifdef ENABLE_GDS
    std::string file_key(filename);
    
    // Check if file already exists
    if (file_resources_.find(file_key) != file_resources_.end()) {
        return true; // Already open
    }
    
    FileResource resource;
    resource.filepath = filename;
    
    // Open file
    resource.fd = open(filename, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (resource.fd < 0) {
        set_error("Failed to open file: " + file_key);
        return false;
    }
    
    // Setup cuFile descriptor
    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(cf_descr));
    cf_descr.handle.fd = resource.fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    // Register file with cuFile
    CUfileError_t status = cuFileHandleRegister(&resource.cf_handle, &cf_descr);
    if (status.err != 0) {
        close(resource.fd);
        set_error("Failed to register file with cuFile: " + file_key);
        return false;
    }
    
    // Store the resource
    file_resources_[file_key] = resource;
    return true;
#else
    set_error("GDS not available");
    return false;
#endif
}

void GDSManager::close_file_internal(const char* filename) {
#ifdef ENABLE_GDS
    std::string file_key(filename);
    auto it = file_resources_.find(file_key);
    if (it != file_resources_.end()) {
        FileResource& resource = it->second;
        
        // Unregister file handle
        cuFileHandleDeregister(resource.cf_handle);
        
        // Close file
        if (resource.fd >= 0) {
            close(resource.fd);
        }
        
        // Remove from map
        file_resources_.erase(it);
    }
#endif
}

GDSManager::FileResource* GDSManager::get_or_create_file_resource(const char* filename) {
#ifdef ENABLE_GDS
    std::string file_key(filename);
    auto it = file_resources_.find(file_key);
    
    if (it != file_resources_.end()) {
        return &(it->second); // Found existing resource
    }
    
    // Create new resource
    if (open_file_internal(filename)) {
        it = file_resources_.find(file_key);
        if (it != file_resources_.end()) {
            return &(it->second);
        }
    }
#endif
    
    return nullptr; // Failed to create or find resource
}

void GDSManager::cleanup() {
#ifdef ENABLE_GDS
    // Synchronize shared stream before cleanup
    if (is_ready_) {
        cudaStreamSynchronize(shared_stream_);
    }
    
    // Destroy any remaining batch handles
    for (auto& pair : batch_info_) {
        cuFileBatchIODestroy(static_cast<CUfileBatchHandle_t>(pair.second.batch_handle));
    }
    batch_info_.clear();
    
    // Close all file resources
    for (auto& pair : file_resources_) {
        FileResource& resource = pair.second;
        
        // Unregister file handle
        cuFileHandleDeregister(resource.cf_handle);
        
        // Close file
        if (resource.fd >= 0) {
            close(resource.fd);
        }
    }
    
    file_resources_.clear();
    
    // Destroy shared CUDA stream
    if (is_ready_) {
        cudaStreamDestroy(shared_stream_);
    }
    
    is_ready_ = false;
#endif
}

size_t GDSManager::get_file_count() const {
#ifdef ENABLE_GDS
    return file_resources_.size();
#else
    return 0;
#endif
}

const std::vector<std::string>& GDSManager::get_file_paths(int device_id) const {
    return file_paths_[device_id];
}

void GDSManager::synchronize() {
#ifdef ENABLE_GDS
    if (is_ready_) {
        cudaStreamSynchronize(shared_stream_);
    }
#endif
}

#ifdef ENABLE_GDS
cudaStream_t GDSManager::get_stream() const {
    return is_ready_ ? shared_stream_ : nullptr;
}
#endif

void GDSManager::initialize_worker_threads() {
#ifdef ENABLE_GDS
    unsigned int hw_threads = std::thread::hardware_concurrency();
    num_worker_threads_ = std::min(hw_threads > 0 ? hw_threads : 4, 8u);
    
    stop_workers_ = false;
    worker_threads_.reserve(num_worker_threads_);
    
    for (int i = 0; i < num_worker_threads_; ++i) {
        worker_threads_.emplace_back([this]() {
            while (true) {
                Task task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    queue_cv_.wait(lock, [this]{ 
                        return stop_workers_ || !task_queue_.empty(); 
                    });
                    
                    if (stop_workers_ && task_queue_.empty()) {
                        return;
                    }
                    
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
                
                task();
            }
        });
    }
#endif
}

void GDSManager::shutdown_worker_threads() {
#ifdef ENABLE_GDS
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_workers_ = true;
    }
    queue_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
#endif
}

std::future<void> GDSManager::enqueue_task(std::function<void()> task) {
#ifdef ENABLE_GDS
    auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
    std::future<void> fut = pkg->get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.emplace([pkg]() { (*pkg)(); });
    }
    queue_cv_.notify_one();
    
    return fut;
#else
    return std::future<void>();
#endif
}

ssize_t GDSManager::write(const char* filename, const void* gpu_data, size_t size, size_t file_offset) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    FileResource* resource = get_or_create_file_resource(filename);
    if (!resource) {
        set_error("Failed to get or create resource for file: " + std::string(filename));
        return -1;
    }
    
#ifdef ENABLE_GDS
    ssize_t bytes_written = cuFileWrite(resource->cf_handle, gpu_data, size, file_offset, 0);
    if (bytes_written < 0) {
        set_error("cuFileWrite failed for file: " + std::string(filename));
        return -1;
    }
    
    return bytes_written;
#else
    return -1;
#endif
}

ssize_t GDSManager::read(const char* filename, void* gpu_buffer, size_t size, size_t file_offset) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    FileResource* resource = get_or_create_file_resource(filename);
    if (!resource) {
        set_error("Failed to get or create resource for file: " + std::string(filename));
        return -1;
    }
    
#ifdef ENABLE_GDS
    ssize_t bytes_read = cuFileRead(resource->cf_handle, gpu_buffer, size, file_offset, 0);
    if (bytes_read < 0) {
        set_error("cuFileRead failed for file: " + std::string(filename));
        return -1;
    }
    
    return bytes_read;
#else
    return -1;
#endif
}

ssize_t GDSManager::write_async(const char* filename, void* gpu_data, size_t size, size_t file_offset) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    FileResource* resource = get_or_create_file_resource(filename);
    if (!resource) {
        set_error("Failed to get or create resource for file: " + std::string(filename));
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Prepare parameters for async write
    size_t write_size = size;
    off_t write_offset = file_offset;
    off_t buffer_offset = 0;
    ssize_t bytes_written = 0;
    
    CUfileError_t status = cuFileWriteAsync(resource->cf_handle, gpu_data, 
                                           &write_size, &write_offset, &buffer_offset, 
                                           &bytes_written, shared_stream_);
    
    if (status.err != 0) {
        set_error("cuFileWriteAsync failed for file: " + std::string(filename));
        return -1;
    }
    
    return bytes_written;
#else
    return -1;
#endif
}

ssize_t GDSManager::read_async(const char* filename, void* gpu_buffer, size_t size, size_t file_offset) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    FileResource* resource = get_or_create_file_resource(filename);
    if (!resource) {
        set_error("Failed to get or create resource for file: " + std::string(filename));
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Prepare parameters for async read
    size_t read_size = size;
    off_t read_offset = file_offset;
    off_t buffer_offset = 0;
    ssize_t bytes_read = 0;
    
    CUfileError_t status = cuFileReadAsync(resource->cf_handle, gpu_buffer, 
                                          &read_size, &read_offset, &buffer_offset, 
                                          &bytes_read, shared_stream_);
    
    if (status.err != 0) {
        set_error("cuFileReadAsync failed for file: " + std::string(filename));
        return -1;
    }
    
    return bytes_read;
#else
    return -1;
#endif
}

int GDSManager::batch_write(const struct BatchWriteOp* operations, int batch_size) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    if (!operations || batch_size <= 0) {
        set_error("Invalid batch write parameters");
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Prepare batch operations
    std::vector<CUfileIOParams_t> io_params(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        const BatchWriteOp& op = operations[i];
        
        if (!op.filename) {
            if (op.result) *op.result = -1;
            continue;
        }
        
        FileResource* resource = get_or_create_file_resource(op.filename);
        if (!resource) {
            if (op.result) *op.result = -1;
            continue;
        }
        
        // Setup IO parameters
        io_params[i].mode = CUFILE_BATCH;
        io_params[i].fh = resource->cf_handle;
        io_params[i].u.batch.devPtr_base = const_cast<void*>(op.gpu_data);
        io_params[i].u.batch.devPtr_offset = 0;
        io_params[i].u.batch.file_offset = op.file_offset;
        io_params[i].u.batch.size = op.size;
        io_params[i].opcode = CUFILE_WRITE;
    }
    
    // Create and submit batch
    CUfileBatchHandle_t batch_handle;
    unsigned int flags = 0;
    CUfileError_t error = cuFileBatchIOSetUp(&batch_handle, batch_size);
    if (error.err != 0) {
        set_error("cuFileBatchIOSetUp failed for batch write");
        return -1;
    }

    CUfileError_t status = cuFileBatchIOSubmit(batch_handle, batch_size, io_params.data(), flags);
    
    if (status.err != 0) {
        set_error("cuFileBatchIOSubmit failed for batch write");
        return -1;
    }
    
    // Store batch info and return batch ID
    int batch_id = next_batch_id_++;
    BatchInfo batch_info;
    batch_info.batch_handle = static_cast<void*>(batch_handle);
    batch_info.batch_size = batch_size;
    batch_info_[batch_id] = batch_info;
    
    return batch_id;
#else
    set_error("GDS not available");
    return -1;
#endif
}

int GDSManager::batch_synchronize(int batch_id) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Find the batch info
    auto it = batch_info_.find(batch_id);
    if (it == batch_info_.end()) {
        set_error("Invalid batch ID");
        return -1;
    }
    
    BatchInfo& batch_info = it->second;
    CUfileBatchHandle_t batch_handle = static_cast<CUfileBatchHandle_t>(batch_info.batch_handle);
    
    // Wait for batch to complete
    unsigned int num_completed = 0;
    unsigned int nr = 0;
    std::vector<CUfileIOEvents_t> io_events(batch_info.batch_size);
    CUfileError_t status;
    
    while (num_completed != static_cast<unsigned int>(batch_info.batch_size)) {
        memset(io_events.data(), 0, io_events.size() * sizeof(CUfileIOEvents_t));
        nr = static_cast<unsigned int>(batch_info.batch_size);
        status = cuFileBatchIOGetStatus(batch_handle, batch_info.batch_size, &nr, io_events.data(), NULL);	
        if (status.err !=0) {
            set_error("cuFileBatchIOGetStatus failed");
            // Clean up the batch handle even if status check failed
            cuFileBatchIODestroy(batch_handle);
            batch_info_.erase(it);
            return -1;
        }

        num_completed += nr;
    } 
    // Destroy the batch
    cuFileBatchIODestroy(batch_handle);
    batch_info_.erase(it);
    
    return 0;
#else
    set_error("GDS not available");
    return -1;
#endif
}

int GDSManager::batch_read(const struct BatchReadOp* operations, int batch_size) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    if (!operations || batch_size <= 0) {
        set_error("Invalid batch read parameters");
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Prepare batch operations
    std::vector<CUfileIOParams_t> io_params(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        const BatchReadOp& op = operations[i];
        
        if (!op.filename) {
            if (op.result) *op.result = -1;
            continue;
        }
        
        FileResource* resource = get_or_create_file_resource(op.filename);
        if (!resource) {
            if (op.result) *op.result = -1;
            continue;
        }
        
        // Setup IO parameters
        io_params[i].mode = CUFILE_BATCH;
        io_params[i].fh = resource->cf_handle;
        io_params[i].u.batch.devPtr_base = op.gpu_buffer;
        io_params[i].u.batch.devPtr_offset = 0;
        io_params[i].u.batch.file_offset = op.file_offset;
        io_params[i].u.batch.size = op.size;
        io_params[i].opcode = CUFILE_READ;
    }
    
    // Create and submit batch
    CUfileBatchHandle_t batch_handle;
    CUfileError_t error = cuFileBatchIOSetUp(&batch_handle, batch_size);
    if (error.err != 0) {
        set_error("cuFileBatchIOSetUp failed for batch read");
        return -1;
    }
    unsigned int flags = 0;
    CUfileError_t status =  cuFileBatchIOSubmit(batch_handle, batch_size, io_params.data(), flags);
    
    if (status.err != 0) {
        set_error("cuFileBatchIOSubmit failed for batch read");
        return -1;
    }
    
    // Store batch info and return batch ID
    int batch_id = next_batch_id_.fetch_add(1);
    BatchInfo batch_info;
    batch_info.batch_handle = static_cast<void*>(batch_handle);
    batch_info.batch_size = batch_size;
    batch_info_[batch_id] = batch_info;
    
    return batch_id;
#else
    set_error("GDS not available");
    return -1;
#endif
}

namespace flexkv {

//partition and remap blocks by device (same logic as SSD transfer)
static void partition_and_remap_blocks_by_device_gds(
    const int64_t* ssd_block_ids, const int64_t* gpu_block_ids, int num_blocks,
    int num_devices, int round_robin,
    std::vector<std::vector<int>>& gpu_blocks_partition,
    std::vector<std::vector<int>>& ssd_blocks_partition) {
    for (int i = 0; i < num_blocks; i++) {
        int64_t ssd_block_id = ssd_block_ids[i];
        int64_t gpu_block_id = gpu_block_ids[i];
        // Use the exact same round-robin mapping as SSD transfer
        int device_id = (ssd_block_id / round_robin) % num_devices;
        int block_id_in_device =
            ((ssd_block_id / round_robin) / num_devices) * round_robin +
            (ssd_block_id % round_robin);
        ssd_blocks_partition[device_id].push_back(block_id_in_device);
        gpu_blocks_partition[device_id].push_back(gpu_block_id);
    }
}

template<BackendType Type>
void transfer_kv_blocks_gds(
    GDSManager& gds_manager,
    const torch::Tensor& gpu_layer_id_list,
    GTensorHandler gpu_tensor_handler,
    const torch::Tensor& ssd_block_ids,
    const torch::Tensor& gpu_block_ids,
    int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int64_t ssd_copy_off_inside_chunks,
    int num_blocks_per_file,
    int64_t total_layers,
    bool is_read,
    bool verbose,
    bool is_mla
) {
    if (!gds_manager.is_ready()) {
        throw std::runtime_error("GDS Manager not ready: " + gds_manager.get_last_error());
    }
    
    int num_devices = gds_manager.get_num_devices();
    int num_files_per_device = gds_manager.get_num_files_per_device();
    int round_robin = gds_manager.get_round_robin();
    
    // Get tensor data pointers
    const int64_t* ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
    const int64_t* gpu_block_id_ptr = gpu_block_ids.data_ptr<int64_t>();
    
    const int num_layers = gpu_layer_id_list.size(0);
    const int32_t* gpu_layer_id_list_ptr = gpu_layer_id_list.data_ptr<int32_t>();
    const int num_transfers = ssd_block_ids.size(0);
    
    // Partition blocks by device using the same logic as SSD
    std::vector<std::vector<int>> gpu_blocks_partition(num_devices, std::vector<int>());
    std::vector<std::vector<int>> ssd_blocks_partition(num_devices, std::vector<int>());
    partition_and_remap_blocks_by_device_gds(
        ssd_block_id_ptr, gpu_block_id_ptr, num_transfers, num_devices, round_robin,
        gpu_blocks_partition, ssd_blocks_partition);
    
    int64_t gpu_layer_stride_in_bytes = gpu_tensor_handler.gpu_layer_stride * sizeof(int64_t);
    int64_t gpu_block_stride_in_bytes = gpu_tensor_handler.gpu_block_stride * sizeof(int64_t);
    
    const bool gpu_is_block_first = gpu_block_stride_in_bytes >= gpu_layer_stride_in_bytes;
    const bool ssd_is_block_first = ssd_block_stride_in_bytes >= ssd_layer_stride_in_bytes;
    const bool enable_block_first_transfer = gpu_is_block_first && ssd_is_block_first;

    std::vector<std::future<void>> transfer_futures;
    
    // Process each device (like SSD transfer)
    for (int device_id = 0; device_id < num_devices; device_id++) {
        const std::vector<int>& gpu_blocks = gpu_blocks_partition[device_id];
        const std::vector<int>& ssd_blocks = ssd_blocks_partition[device_id];
        
        const std::vector<std::string>& file_list = gds_manager.get_file_paths(device_id);
        
        // Submit task to the thread pool
        for (size_t j = 0; j < gpu_blocks.size(); j++) {
            transfer_futures.push_back(gds_manager.enqueue_task([&, device_id, j]() {
                int64_t gpu_block_id = gpu_blocks[j];
                int64_t ssd_block_id = ssd_blocks[j];
                
                int file_id_in_device = ssd_block_id % num_files_per_device;
                const std::string& filename = file_list[file_id_in_device];
                int64_t block_id_in_file = ssd_block_id / num_files_per_device;
                
                // Block-first transfer
                if (enable_block_first_transfer) {
                    int32_t start_layer = gpu_layer_id_list_ptr[0];
                    int64_t layers_chunk_size_in_bytes = gpu_layer_stride_in_bytes * num_layers;
                    int64_t ssd_layers_chunk_offset = start_layer * ssd_layer_stride_in_bytes;
                    
                    int64_t *gpu_block_ptr = ptr_at<Type>(gpu_tensor_handler, 0, 0, gpu_block_id);
                    
                    int64_t ssd_block_offset = 
                        ssd_block_stride_in_bytes * block_id_in_file + 
                        ssd_layers_chunk_offset + 
                        ssd_copy_off_inside_chunks;
                    
                    ssize_t transfer_result;
                    if (is_read) {
                        // SSD -> GPU
                        transfer_result = gds_manager.read(filename.c_str(), gpu_block_ptr, 
                                                          layers_chunk_size_in_bytes, ssd_block_offset);
                    } else {
                        // GPU -> SSD
                        transfer_result = gds_manager.write(filename.c_str(), gpu_block_ptr, 
                                                           layers_chunk_size_in_bytes, ssd_block_offset);
                    }
                    
                    if (transfer_result != layers_chunk_size_in_bytes) {
                        throw std::runtime_error("Failed to transfer block " + std::to_string(j) +
                                               " (block-first mode), file " + filename + 
                                               ", size=" + std::to_string(layers_chunk_size_in_bytes) +
                                               ", offset=" + std::to_string(ssd_block_offset) +
                                               ": " + gds_manager.get_last_error());
                    }
                    
                    if (verbose) {
                        std::cerr << "Block " << j << " (block-first)"
                                  << " Operation: " << (is_read ? "Read" : "Write")
                                  << " Device: " << device_id 
                                  << " File_in_device: " << file_id_in_device
                                  << " Block_in_file: " << block_id_in_file
                                  << " GPU Block ID: " << gpu_block_id 
                                  << " Layers: " << num_layers
                                  << " Total bytes: " << transfer_result << std::endl;
                    }
                    
                    return;
                }
                
                // Layer-wise transfer
                for (int i = 0; i < num_layers; i++) {
                    int32_t layer_idx = gpu_layer_id_list_ptr[i];
                    
                    int64_t *gpu_k_ptr = ptr_at<Type>(gpu_tensor_handler, i, 0, gpu_block_id);
                    int64_t *gpu_v_ptr = is_mla ? nullptr : ptr_at<Type>(gpu_tensor_handler, i, 1, gpu_block_id);
                    
                    int64_t ssd_base_offset = 
                        ssd_layer_stride_in_bytes * layer_idx +
                        ssd_block_stride_in_bytes * block_id_in_file;
                    
                    int64_t ssd_k_offset = ssd_base_offset + ssd_copy_off_inside_chunks;
                    int64_t ssd_v_offset = ssd_k_offset + ssd_kv_stride_in_bytes;
                    
                    ssize_t k_result;
                    if (is_read) {
                        // SSD -> GPU (read from SSD file to GPU memory)
                        k_result = gds_manager.read(filename.c_str(), gpu_k_ptr, chunk_size_in_bytes, ssd_k_offset);
                    } else {
                        // GPU -> SSD (write from GPU memory to SSD file) 
                        k_result = gds_manager.write(filename.c_str(), gpu_k_ptr, chunk_size_in_bytes, ssd_k_offset);
                    }
                    
                    if (k_result != chunk_size_in_bytes) {
                        throw std::runtime_error("Failed to transfer K block for layer " + 
                                               std::to_string(layer_idx) + ", block " + std::to_string(j) +
                                               ", file " + filename + ": " + gds_manager.get_last_error());
                    }
                    
                    if (is_mla) {
                        if (verbose) {
                            std::cerr << "Layer " << layer_idx << " Block " << j
                                      << " Operation: " << (is_read ? "Read" : "Write")
                                      << " Device: " << device_id 
                                      << " File_in_device: " << file_id_in_device
                                      << " Block_in_file: " << block_id_in_file
                                      << " GPU Block ID: " << gpu_block_id 
                                      << " K bytes: " << k_result << std::endl;
                        }
                        continue;
                    }

                    ssize_t v_result;
                    if (is_read) {
                        v_result = gds_manager.read(filename.c_str(), gpu_v_ptr, chunk_size_in_bytes, ssd_v_offset);
                    } else {
                        v_result = gds_manager.write(filename.c_str(), gpu_v_ptr, chunk_size_in_bytes, ssd_v_offset);
                    }
                    
                    if (v_result != chunk_size_in_bytes) {
                        throw std::runtime_error("Failed to transfer V block for layer " + 
                                               std::to_string(layer_idx) + ", block " + std::to_string(j) +
                                               ", file " + filename + ": " + gds_manager.get_last_error());
                    }
                    
                    if (verbose) {
                        std::cerr << "Layer " << layer_idx << " Block " << j
                                  << " Operation: " << (is_read ? "Read" : "Write")
                                  << " Device: " << device_id 
                                  << " File_in_device: " << file_id_in_device
                                  << " Block_in_file: " << block_id_in_file
                                  << " GPU Block ID: " << gpu_block_id 
                                  << " K bytes: " << k_result
                                  << " V bytes: " << v_result << std::endl;
                    }
                } // end layer loop
            })); // enqueue_task
        } // end block loop
    } // end device loop
    
    for (auto& future : transfer_futures) {
        try {
            future.get();
        } catch (const std::exception& e) {
            for (auto& remaining_future : transfer_futures) {
                try {
                    remaining_future.wait();
                } catch (...) {
                }
            }
            throw;
        }
    }
    // Synchronize to ensure all operations complete
    gds_manager.synchronize();
}

// Explicit template instantiations
template void transfer_kv_blocks_gds<BackendType::VLLM>(
    GDSManager&, const torch::Tensor&, GTensorHandler, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int64_t, bool, bool, bool);

template void transfer_kv_blocks_gds<BackendType::TRTLLM>(
    GDSManager&, const torch::Tensor&, GTensorHandler, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int64_t, bool, bool, bool);

template void transfer_kv_blocks_gds<BackendType::SGLANG>(
    GDSManager&, const torch::Tensor&, GTensorHandler, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int64_t, bool, bool, bool);

/**
 * Staged GDS transfer implementation
 * 
 * For non-TRTLLM backends (VLLM, SGLANG), this function:
 * 1. Allocates a staging buffer with block-first layout (same as SSD)
 * 2. For READ: GDS transfer SSD -> staging buffer, then CUDA kernel transform staging -> target GPU
 * 3. For WRITE: CUDA kernel transform target GPU -> staging, then GDS transfer staging -> SSD
 * 
 * For TRTLLM backend, directly uses block-first transfer (no staging needed).
 */
template<BackendType Type>
void transfer_kv_blocks_gds_staged(
    GDSManager& gds_manager,
    const torch::Tensor& gpu_layer_id_list,
    GTensorHandler gpu_tensor_handler,
    const torch::Tensor& ssd_block_ids,
    const torch::Tensor& gpu_block_ids,
    int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int64_t ssd_copy_off_inside_chunks,
    int num_blocks_per_file,
    int64_t total_layers,
    bool is_read,
    void* staging_buffer,
    bool verbose,
    bool is_mla
) {
    if (!gds_manager.is_ready()) {
        throw std::runtime_error("GDS Manager not ready: " + gds_manager.get_last_error());
    }
    
    // For TRTLLM, GPU layout is already block-first, use direct transfer
    if constexpr (Type == BackendType::TRTLLM) {
        transfer_kv_blocks_gds<Type>(
            gds_manager, gpu_layer_id_list, gpu_tensor_handler,
            ssd_block_ids, gpu_block_ids,
            ssd_layer_stride_in_bytes, ssd_block_stride_in_bytes, ssd_kv_stride_in_bytes,
            chunk_size_in_bytes, ssd_copy_off_inside_chunks,
            num_blocks_per_file, total_layers, is_read, verbose, is_mla);
        return;
    }
    
    // For VLLM/SGLANG, use staging buffer approach
    int num_devices = gds_manager.get_num_devices();
    int num_files_per_device = gds_manager.get_num_files_per_device();
    int round_robin = gds_manager.get_round_robin();
    
    const int64_t* ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
    const int64_t* gpu_block_id_ptr = gpu_block_ids.data_ptr<int64_t>();
    
    const int num_layers = gpu_layer_id_list.size(0);
    const int32_t* gpu_layer_id_list_ptr = gpu_layer_id_list.data_ptr<int32_t>();
    const int num_transfers = ssd_block_ids.size(0);
    
    if (num_transfers == 0) return;
    
    // Calculate staging buffer size (block-first layout)
    // Staging layout matches SSD: [block][layer][kv][data]
    int kv_dim = is_mla ? 1 : 2;
    int64_t staging_layer_stride_in_bytes = chunk_size_in_bytes * kv_dim;
    int64_t staging_block_stride_in_bytes = staging_layer_stride_in_bytes * num_layers;
    int64_t staging_kv_stride_in_bytes = chunk_size_in_bytes;
    int64_t total_staging_size = staging_block_stride_in_bytes * num_transfers;
    
    // Allocate or use provided staging buffer
    void* staging_ptr = staging_buffer;
    bool own_staging_buffer = false;
    
    if (!staging_ptr) {
#ifdef ENABLE_GDS
        cudaError_t cuda_status = cudaMalloc(&staging_ptr, total_staging_size);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate staging buffer: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }
        own_staging_buffer = true;
#else
        throw std::runtime_error("GDS not available");
#endif
    }
    
#ifdef ENABLE_GDS
    // Get CUDA stream for layout transform
    cudaStream_t transform_stream = gds_manager.get_stream();
    
    // Convert strides to int64_t units for kernel
    int64_t staging_layer_stride = staging_layer_stride_in_bytes / sizeof(int64_t);
    int64_t staging_kv_stride = staging_kv_stride_in_bytes / sizeof(int64_t);
    int64_t staging_block_stride = staging_block_stride_in_bytes / sizeof(int64_t);
    int64_t chunk_size = chunk_size_in_bytes / sizeof(int64_t);
    
    // Partition blocks by device
    std::vector<std::vector<int>> gpu_blocks_partition(num_devices);
    std::vector<std::vector<int>> ssd_blocks_partition(num_devices);
    std::vector<std::vector<int>> staging_idx_partition(num_devices);  // Index in staging buffer
    
    for (int i = 0; i < num_transfers; i++) {
        int64_t ssd_block_id = ssd_block_id_ptr[i];
        int64_t gpu_block_id = gpu_block_id_ptr[i];
        int device_id = (ssd_block_id / round_robin) % num_devices;
        int block_id_in_device =
            ((ssd_block_id / round_robin) / num_devices) * round_robin +
            (ssd_block_id % round_robin);
        ssd_blocks_partition[device_id].push_back(block_id_in_device);
        gpu_blocks_partition[device_id].push_back(gpu_block_id);
        staging_idx_partition[device_id].push_back(i);
    }
    
    // For WRITE operation: first transform GPU -> staging
    if (!is_read) {
        // Prepare gpu_block_ids on device for kernel
        int64_t* d_gpu_block_ids;
        cudaMalloc(&d_gpu_block_ids, num_transfers * sizeof(int64_t));
        cudaMemcpy(d_gpu_block_ids, gpu_block_id_ptr, num_transfers * sizeof(int64_t), 
                   cudaMemcpyHostToDevice);
        
        // Transform from target GPU tensor to staging buffer
        launch_layout_transform_kernel<Type>(
            reinterpret_cast<int64_t*>(staging_ptr),
            staging_layer_stride,
            staging_kv_stride,
            staging_block_stride,
            chunk_size,
            gpu_tensor_handler,
            d_gpu_block_ids,
            num_transfers,
            num_layers,
            is_mla,
            false,  // staging_to_target = false (GPU -> staging)
            transform_stream
        );
        
        cudaStreamSynchronize(transform_stream);
        cudaFree(d_gpu_block_ids);
    }
    
    // GDS transfer: staging <-> SSD (block-first, highly efficient)
    std::vector<std::future<void>> transfer_futures;
    
    for (int device_id = 0; device_id < num_devices; device_id++) {
        const std::vector<int>& gpu_blocks = gpu_blocks_partition[device_id];
        const std::vector<int>& ssd_blocks = ssd_blocks_partition[device_id];
        const std::vector<int>& staging_indices = staging_idx_partition[device_id];
        
        const std::vector<std::string>& file_list = gds_manager.get_file_paths(device_id);
        
        for (size_t j = 0; j < ssd_blocks.size(); j++) {
            transfer_futures.push_back(gds_manager.enqueue_task([&, device_id, j, 
                                        staging_ptr, staging_block_stride_in_bytes,
                                        ssd_copy_off_inside_chunks]() {
                int staging_idx = staging_indices[j];
                int64_t ssd_block_id = ssd_blocks[j];
                
                int file_id_in_device = ssd_block_id % num_files_per_device;
                const std::string& filename = file_list[file_id_in_device];
                int64_t block_id_in_file = ssd_block_id / num_files_per_device;
                
                // Calculate pointers
                int32_t start_layer = gpu_layer_id_list_ptr[0];
                int64_t layers_chunk_size_in_bytes = staging_block_stride_in_bytes;
                
                char* staging_block_ptr = reinterpret_cast<char*>(staging_ptr) + 
                                         staging_idx * staging_block_stride_in_bytes;
                
                int64_t ssd_block_offset = 
                    ssd_block_stride_in_bytes * block_id_in_file + 
                    start_layer * ssd_layer_stride_in_bytes +
                    ssd_copy_off_inside_chunks;
                
                ssize_t transfer_result;
                if (is_read) {
                    transfer_result = gds_manager.read(filename.c_str(), staging_block_ptr,
                                                      layers_chunk_size_in_bytes, ssd_block_offset);
                } else {
                    transfer_result = gds_manager.write(filename.c_str(), staging_block_ptr,
                                                       layers_chunk_size_in_bytes, ssd_block_offset);
                }
                
                if (transfer_result != layers_chunk_size_in_bytes) {
                    throw std::runtime_error("GDS transfer failed for block " + std::to_string(j) +
                                           ", file " + filename + 
                                           ", expected=" + std::to_string(layers_chunk_size_in_bytes) +
                                           ", got=" + std::to_string(transfer_result) +
                                           ": " + gds_manager.get_last_error());
                }
                
                if (verbose) {
                    std::cerr << "Staged transfer - Block " << j 
                              << " Operation: " << (is_read ? "Read" : "Write")
                              << " Device: " << device_id
                              << " File_in_device: " << file_id_in_device
                              << " Block_in_file: " << block_id_in_file
                              << " Staging_idx: " << staging_idx
                              << " Total bytes: " << transfer_result << std::endl;
                }
            }));
        }
    }
    
    // Wait for all GDS transfers to complete
    for (auto& future : transfer_futures) {
        try {
            future.get();
        } catch (const std::exception& e) {
            for (auto& remaining_future : transfer_futures) {
                try {
                    remaining_future.wait();
                } catch (...) {}
            }
            if (own_staging_buffer) {
                cudaFree(staging_ptr);
            }
            throw;
        }
    }
    
    // For READ operation: transform staging -> GPU
    if (is_read) {
        // Prepare gpu_block_ids on device for kernel
        int64_t* d_gpu_block_ids;
        cudaMalloc(&d_gpu_block_ids, num_transfers * sizeof(int64_t));
        cudaMemcpy(d_gpu_block_ids, gpu_block_id_ptr, num_transfers * sizeof(int64_t), 
                   cudaMemcpyHostToDevice);
        
        // Transform from staging buffer to target GPU tensor
        launch_layout_transform_kernel<Type>(
            reinterpret_cast<int64_t*>(staging_ptr),
            staging_layer_stride,
            staging_kv_stride,
            staging_block_stride,
            chunk_size,
            gpu_tensor_handler,
            d_gpu_block_ids,
            num_transfers,
            num_layers,
            is_mla,
            true,  // staging_to_target = true (staging -> GPU)
            transform_stream
        );
        
        cudaStreamSynchronize(transform_stream);
        cudaFree(d_gpu_block_ids);
    }
    
    // Cleanup
    if (own_staging_buffer) {
        cudaFree(staging_ptr);
    }
    
    gds_manager.synchronize();
#else
    (void)staging_ptr;
    (void)own_staging_buffer;
    throw std::runtime_error("GDS not available");
#endif
}

// Explicit template instantiations for staged transfer
template void transfer_kv_blocks_gds_staged<BackendType::VLLM>(
    GDSManager&, const torch::Tensor&, GTensorHandler, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int64_t, bool, void*, bool, bool);

template void transfer_kv_blocks_gds_staged<BackendType::TRTLLM>(
    GDSManager&, const torch::Tensor&, GTensorHandler, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int64_t, bool, void*, bool, bool);

template void transfer_kv_blocks_gds_staged<BackendType::SGLANG>(
    GDSManager&, const torch::Tensor&, GTensorHandler, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int64_t, bool, void*, bool, bool);

}