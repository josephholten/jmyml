// TODO: variadic template parameters: this allows us to have Layer not be a virtual class, decreasing overhead and incresing speed (?)
template<class... Layers>
class Sequential {
    void forward();
    // void backward();

private:
    size_t in_dim;
    size_t out_dim;
    std::tuple<Layers...> layers;
};