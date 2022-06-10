#include "reef/client/client.h"
#include <thread>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << std::string(argv[0]) << " model_dir model_name real_time sleep_time(ms)\n";
        std::cerr << "Example: " << std::string(argv[0]) << " reef/resource/resnet18 resnet18 1 10\n";
        return -1;
    }

    std::string model_dir(argv[1]);
    std::string model_name(argv[2]);
    int real_time = std::atoi(argv[3]);
    int sleep_time = std::atoi(argv[4]);
    

    reef::client::REEFClient client(DEFAULT_REEF_ADDR);
    ASSERT(client.init(real_time)); // whether this client send real-time requests?
    
    std::cout << "loading '" << model_name << "' from " << "'"<< model_dir << "'\n";
    auto model = client.load_model(model_dir, model_name);
    ASSERT(model.get() != nullptr);

    // Get or set the input/output data.
    // auto input_blob = model->get_input_blob();
    // model->load_input();
    // auto output_blob = model->get_output_blob();
    // auto output = model->get_output();

    std::cout << "submit inference requests\n";
    while (true) {
        auto task = model->infer(); // submit an inference request
        std::cout << "client " << model->get_mid() << " inference latency: " << std::chrono::duration_cast<std::chrono::microseconds>(task.finish - task.submit).count() / 1000.0 << " ms\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }

    return 0;
}