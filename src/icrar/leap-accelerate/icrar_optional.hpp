
#if (__cplusplus >= 201700L)
#include <optional>
namespace icrar
{
    template<class T>
    using optional = std::optional<T>;
}
#else
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
namespace icrar
{
    template<class T>
    using optional = boost::optional<T>;
}
#endif