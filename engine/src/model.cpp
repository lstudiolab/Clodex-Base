#include "clodex/model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kVocabulary = 258;
constexpr int kFormatVersion = 1;
constexpr char kMagic[8] = {'C','L','O','D','E','X','1','\0'};
thread_local std::string g_last_error;

struct Parameters {
    int hidden = 128;
    std::uint64_t updates = 0;
    std::vector<float> embedding, recurrent, output, output_bias;
    std::vector<float> m_embedding, v_embedding, m_recurrent, v_recurrent;
    std::vector<float> m_output, v_output, m_bias, v_bias;

    explicit Parameters(int width) : hidden(width) {
        const auto vh = static_cast<std::size_t>(kVocabulary) * hidden;
        const auto hh = static_cast<std::size_t>(hidden) * hidden;
        embedding.resize(vh); recurrent.resize(hh); output.resize(vh);
        output_bias.assign(kVocabulary, 0.0f);
        m_embedding.assign(vh, 0.0f); v_embedding.assign(vh, 0.0f);
        m_recurrent.assign(hh, 0.0f); v_recurrent.assign(hh, 0.0f);
        m_output.assign(vh, 0.0f); v_output.assign(vh, 0.0f);
        m_bias.assign(kVocabulary, 0.0f); v_bias.assign(kVocabulary, 0.0f);
    }
};

struct Gradients {
    std::vector<float> embedding, recurrent, output, bias;
    explicit Gradients(const Parameters& p)
        : embedding(p.embedding.size(), 0.0f),
          recurrent(p.recurrent.size(), 0.0f),
          output(p.output.size(), 0.0f),
          bias(kVocabulary, 0.0f) {}
    void clear() {
        std::fill(embedding.begin(), embedding.end(), 0.0f);
        std::fill(recurrent.begin(), recurrent.end(), 0.0f);
        std::fill(output.begin(), output.end(), 0.0f);
        std::fill(bias.begin(), bias.end(), 0.0f);
    }
};

struct Frame {
    int input = 0;
    int target = 0;
    std::vector<float> previous;
    std::vector<float> state;
    std::vector<float> probabilities;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_corpus(const char* path) {
    require(path != nullptr, "corpus path is null");
    std::ifstream file(path, std::ios::binary);
    require(file.good(), std::string("cannot open corpus: ") + path);
    std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    require(bytes.size() >= 4, "corpus must contain at least four bytes");
    return bytes;
}

void initialize(Parameters& p, std::mt19937_64& rng) {
    std::normal_distribution<float> normal(0.0f, 0.04f);
    for (auto& x : p.embedding) x = normal(rng);
    const float scale = 1.0f / std::sqrt(static_cast<float>(p.hidden));
    for (auto& x : p.recurrent) x = normal(rng) * scale;
    for (auto& x : p.output) x = normal(rng) * scale;
}

template <typename T>
void write_value(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    require(out.good(), "failed writing checkpoint");
}
template <typename T>
void read_value(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    require(in.good(), "checkpoint is truncated or unreadable");
}
void write_vector(std::ofstream& out, const std::vector<float>& values) {
    const std::uint64_t n = values.size();
    write_value(out, n);
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    require(out.good(), "failed writing checkpoint tensor");
}
void read_vector(std::ifstream& in, std::vector<float>& values, std::size_t expected) {
    std::uint64_t n = 0;
    read_value(in, n);
    require(n == expected, "checkpoint tensor dimensions do not match selected architecture");
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    require(in.good(), "checkpoint tensor is truncated");
}

void save_checkpoint(const std::string& path, const Parameters& p) {
    namespace fs = std::filesystem;
    const fs::path target(path);
    if (!target.parent_path().empty()) fs::create_directories(target.parent_path());
    const fs::path temp = target.string() + ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        require(out.good(), "cannot open temporary checkpoint for writing");
        out.write(kMagic, sizeof(kMagic));
        write_value(out, kFormatVersion);
        write_value(out, kVocabulary);
        write_value(out, p.hidden);
        write_value(out, p.updates);
        write_vector(out, p.embedding); write_vector(out, p.recurrent);
        write_vector(out, p.output); write_vector(out, p.output_bias);
        write_vector(out, p.m_embedding); write_vector(out, p.v_embedding);
        write_vector(out, p.m_recurrent); write_vector(out, p.v_recurrent);
        write_vector(out, p.m_output); write_vector(out, p.v_output);
        write_vector(out, p.m_bias); write_vector(out, p.v_bias);
        out.flush();
        require(out.good(), "failed flushing checkpoint");
    }
    std::error_code ec;
    fs::rename(temp, target, ec);
    if (ec) {
        fs::remove(target, ec);
        ec.clear();
        fs::rename(temp, target, ec);
    }
    require(!ec, std::string("cannot replace checkpoint: ") + ec.message());
}

int checkpoint_hidden_size(const char* path) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "cannot open checkpoint");
    char magic[sizeof(kMagic)]{};
    in.read(magic, sizeof(magic));
    require(in.good() && std::memcmp(magic, kMagic, sizeof(kMagic)) == 0, "checkpoint magic/version signature is invalid");
    int version = 0, vocab = 0, hidden = 0;
    read_value(in, version); read_value(in, vocab); read_value(in, hidden);
    require(version == kFormatVersion && vocab == kVocabulary, "unsupported checkpoint architecture");
    require(hidden >= 8 && hidden <= 2048, "checkpoint hidden size is outside supported range");
    return hidden;
}

void load_checkpoint(const std::string& path, Parameters& p) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "cannot open checkpoint");
    char magic[sizeof(kMagic)]{};
    in.read(magic, sizeof(magic));
    require(in.good() && std::memcmp(magic, kMagic, sizeof(kMagic)) == 0, "checkpoint magic/version signature is invalid");
    int version = 0, vocab = 0, hidden = 0;
    read_value(in, version); read_value(in, vocab); read_value(in, hidden); read_value(in, p.updates);
    require(version == kFormatVersion, "unsupported checkpoint format version");
    require(vocab == kVocabulary, "checkpoint tokenizer vocabulary is incompatible");
    require(hidden == p.hidden, "hidden size differs from checkpoint; resume using the original --hidden-size");
    read_vector(in, p.embedding, p.embedding.size()); read_vector(in, p.recurrent, p.recurrent.size());
    read_vector(in, p.output, p.output.size()); read_vector(in, p.output_bias, p.output_bias.size());
    read_vector(in, p.m_embedding, p.m_embedding.size()); read_vector(in, p.v_embedding, p.v_embedding.size());
    read_vector(in, p.m_recurrent, p.m_recurrent.size()); read_vector(in, p.v_recurrent, p.v_recurrent.size());
    read_vector(in, p.m_output, p.m_output.size()); read_vector(in, p.v_output, p.v_output.size());
    read_vector(in, p.m_bias, p.m_bias.size()); read_vector(in, p.v_bias, p.v_bias.size());
}

void softmax(const Parameters& p, const std::vector<float>& state, float temperature, std::vector<float>& probs) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < kVocabulary; ++token) {
        float logit = p.output_bias[token];
        const std::size_t base = static_cast<std::size_t>(token) * p.hidden;
        for (int j = 0; j < p.hidden; ++j) logit += p.output[base + j] * state[j];
        probs[token] = logit / temperature;
        max_logit = std::max(max_logit, probs[token]);
    }
    double sum = 0.0;
    for (auto& value : probs) { value = std::exp(value - max_logit); sum += value; }
    const float denom = static_cast<float>(sum > 0.0 ? sum : 1.0);
    for (auto& value : probs) value /= denom;
}

void forward_one(const Parameters& p, int input, const std::vector<float>& previous,
                 std::vector<float>& state) {
    const std::size_t ebase = static_cast<std::size_t>(input) * p.hidden;
    for (int j = 0; j < p.hidden; ++j) {
        float value = p.embedding[ebase + j];
        const std::size_t row = static_cast<std::size_t>(j) * p.hidden;
        for (int k = 0; k < p.hidden; ++k) value += p.recurrent[row + k] * previous[k];
        state[j] = std::tanh(value);
    }
}

void adam_update(std::vector<float>& weights, const std::vector<float>& grad,
                 std::vector<float>& first, std::vector<float>& second,
                 std::uint64_t step, float learning_rate) {
    constexpr float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1.0e-8f;
    const float correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    for (std::size_t i = 0; i < weights.size(); ++i) {
        first[i] = beta1 * first[i] + (1.0f - beta1) * grad[i];
        second[i] = beta2 * second[i] + (1.0f - beta2) * grad[i] * grad[i];
        const float mhat = first[i] / correction1;
        const float vhat = second[i] / correction2;
        weights[i] -= learning_rate * mhat / (std::sqrt(vhat) + epsilon);
    }
}

float train_step(Parameters& p, Gradients& grad, const std::string& corpus,
                 int sequence_length, std::mt19937_64& rng, float learning_rate) {
    const std::size_t max_start = corpus.size() - static_cast<std::size_t>(sequence_length) - 1;
    std::uniform_int_distribution<std::size_t> pick(0, max_start);
    const std::size_t start = pick(rng);
    std::vector<Frame> frames(static_cast<std::size_t>(sequence_length));
    std::vector<float> state(p.hidden, 0.0f);
    std::vector<float> zero(p.hidden, 0.0f);
    std::vector<float> probs(kVocabulary, 0.0f);
    double loss = 0.0;

    for (int t = 0; t < sequence_length; ++t) {
        Frame& f = frames[static_cast<std::size_t>(t)];
        f.input = static_cast<unsigned char>(corpus[start + static_cast<std::size_t>(t)]);
        f.target = static_cast<unsigned char>(corpus[start + static_cast<std::size_t>(t) + 1]);
        f.previous = (t == 0) ? zero : state;
        f.state.assign(p.hidden, 0.0f);
        forward_one(p, f.input, f.previous, f.state);
        f.probabilities.assign(kVocabulary, 0.0f);
        softmax(p, f.state, 1.0f, f.probabilities);
        const float chosen = std::max(f.probabilities[f.target], 1.0e-12f);
        loss -= std::log(chosen);
        state = f.state;
    }

    grad.clear();
    std::vector<float> dh(p.hidden, 0.0f);
    std::vector<float> dprev(p.hidden, 0.0f);
    for (int t = sequence_length - 1; t >= 0; --t) {
        const Frame& f = frames[static_cast<std::size_t>(t)];
        std::vector<float> dlogits = f.probabilities;
        dlogits[f.target] -= 1.0f;
        const float inv_sequence = 1.0f / static_cast<float>(sequence_length);
        for (int token = 0; token < kVocabulary; ++token) {
            const float g = dlogits[token] * inv_sequence;
            grad.bias[token] += g;
            const std::size_t base = static_cast<std::size_t>(token) * p.hidden;
            for (int j = 0; j < p.hidden; ++j) grad.output[base + j] += g * f.state[j];
        }
        std::fill(dh.begin(), dh.end(), 0.0f);
        for (int j = 0; j < p.hidden; ++j) {
            for (int token = 0; token < kVocabulary; ++token) {
                dh[j] += p.output[static_cast<std::size_t>(token) * p.hidden + j] * dlogits[token] * inv_sequence;
            }
        }
        for (int j = 0; j < p.hidden; ++j) dh[j] *= (1.0f - f.state[j] * f.state[j]);
        const std::size_t ebase = static_cast<std::size_t>(f.input) * p.hidden;
        for (int j = 0; j < p.hidden; ++j) grad.embedding[ebase + j] += dh[j];
        std::fill(dprev.begin(), dprev.end(), 0.0f);
        for (int i = 0; i < p.hidden; ++i) {
            const std::size_t row = static_cast<std::size_t>(i) * p.hidden;
            for (int j = 0; j < p.hidden; ++j) {
                grad.recurrent[row + j] += dh[i] * f.previous[j];
                dprev[j] += p.recurrent[row + j] * dh[i];
            }
        }
        dh.swap(dprev);
    }

    double norm_sq = 0.0;
    auto accumulate_norm = [&norm_sq](const std::vector<float>& g) {
        for (float value : g) norm_sq += static_cast<double>(value) * value;
    };
    accumulate_norm(grad.embedding); accumulate_norm(grad.recurrent);
    accumulate_norm(grad.output); accumulate_norm(grad.bias);
    const float norm = static_cast<float>(std::sqrt(norm_sq));
    if (norm > 1.0f) {
        const float scale = 1.0f / norm;
        for (auto& x : grad.embedding) x *= scale;
        for (auto& x : grad.recurrent) x *= scale;
        for (auto& x : grad.output) x *= scale;
        for (auto& x : grad.bias) x *= scale;
    }

    ++p.updates;
    adam_update(p.embedding, grad.embedding, p.m_embedding, p.v_embedding, p.updates, learning_rate);
    adam_update(p.recurrent, grad.recurrent, p.m_recurrent, p.v_recurrent, p.updates, learning_rate);
    adam_update(p.output, grad.output, p.m_output, p.v_output, p.updates, learning_rate);
    adam_update(p.output_bias, grad.bias, p.m_bias, p.v_bias, p.updates, learning_rate);
    return static_cast<float>(loss / sequence_length);
}

int sample_token(const std::vector<float>& probs, std::mt19937_64& rng) {
    std::discrete_distribution<int> distribution(probs.begin(), probs.end());
    return distribution(rng);
}
} // namespace

extern "C" const char* clodex_engine_version() {
    return "clodex-cpu-rnn/0.1.0";
}
extern "C" const char* clodex_last_error() { return g_last_error.c_str(); }

extern "C" int clodex_train(const char* corpus_path, const char* checkpoint_path, int steps,
                            int sequence_length, int hidden_size, double learning_rate,
                            int resume, int save_every) {
    try {
        require(checkpoint_path != nullptr, "checkpoint path is null");
        require(steps > 0 && sequence_length >= 2 && hidden_size >= 8 && hidden_size <= 2048,
                "invalid training dimensions");
        require(save_every > 0, "save interval must be positive");
        require(std::isfinite(learning_rate) && learning_rate > 0.0 && learning_rate <= 1.0,
                "learning rate is invalid");
        const std::string corpus = read_corpus(corpus_path);
        require(corpus.size() > static_cast<std::size_t>(sequence_length) + 1,
                "corpus is too short for the requested sequence length");
        Parameters model(hidden_size);
        std::mt19937_64 rng(0xC10D3ULL);
        bool resumed = false;
        if (resume && std::filesystem::is_regular_file(checkpoint_path)) {
            load_checkpoint(checkpoint_path, model);
            resumed = true;
            rng.seed(0xC10D3ULL + model.updates);
        } else {
            initialize(model, rng);
        }
        std::printf("mode=%s updates=%llu hidden=%d vocab=%d sequence=%d\n",
                    resumed ? "resume" : "fresh",
                    static_cast<unsigned long long>(model.updates), model.hidden, kVocabulary, sequence_length);
        Gradients gradients(model);
        for (int i = 0; i < steps; ++i) {
            const float loss = train_step(model, gradients, corpus, sequence_length, rng, static_cast<float>(learning_rate));
            if ((i + 1) % 10 == 0 || i == 0) {
                std::printf("update=%llu session=%d/%d loss=%.4f\n",
                            static_cast<unsigned long long>(model.updates), i + 1, steps, loss);
                std::fflush(stdout);
            }
            if ((i + 1) % save_every == 0 || i + 1 == steps) {
                save_checkpoint(checkpoint_path, model);
            }
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 1;
    }
}

extern "C" int clodex_generate(const char* checkpoint_path, const unsigned char* prompt,
                               std::size_t prompt_size, std::size_t max_new_tokens,
                               std::uint64_t seed, double temperature,
                               unsigned char* output, std::size_t output_capacity,
                               std::size_t* output_size) {
    try {
        require(checkpoint_path != nullptr, "checkpoint path is null");
        require(prompt != nullptr || prompt_size == 0, "prompt pointer is null");
        require(output != nullptr || output_capacity == 0, "output pointer is null");
        require(output_size != nullptr, "output size pointer is null");
        require(output_capacity >= max_new_tokens, "output buffer is too small");
        require(max_new_tokens <= 16384, "generation length exceeds safety limit");
        require(std::isfinite(temperature) && temperature >= 0.05 && temperature <= 3.0,
                "temperature must be in [0.05, 3.0]");
        Parameters model(checkpoint_hidden_size(checkpoint_path));
        load_checkpoint(checkpoint_path, model);
        *output_size = 0;
        std::vector<float> state(model.hidden, 0.0f), next(model.hidden, 0.0f), probs(kVocabulary, 0.0f);
        for (std::size_t i = 0; i < prompt_size; ++i) {
            forward_one(model, prompt[i], state, next);
            state.swap(next);
        }
        std::mt19937_64 rng(seed);
        for (std::size_t i = 0; i < max_new_tokens; ++i) {
            softmax(model, state, static_cast<float>(temperature), probs);
            const int token = sample_token(probs, rng);
            // The baseline emits bytes only; reserved BOS/EOS IDs are not sampled as text.
            if (token > 255) {
                output[i] = static_cast<unsigned char>(' ');
            } else {
                output[i] = static_cast<unsigned char>(token);
            }
            *output_size = i + 1;
            forward_one(model, token <= 255 ? token : static_cast<int>(' '), state, next);
            state.swap(next);
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 1;
    }
}
