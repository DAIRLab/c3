#include "multibody/contact_evaluator.h"

namespace c3 {
namespace multibody {

// Explicit template instantiations for double precision
template class ContactEvaluator<double>;
template class PolytopeContactEvaluator<double>;
template class PlanarContactEvaluator<double>;

}  // namespace multibody
}  // namespace c3