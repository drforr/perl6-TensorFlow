use TensorFlow::Graph;

my role Debugging {
}

my role Testing {

}

my role Validating {

}

class TensorFlow::Tensor {
	also does Debugging;
	also does Testing;
	also does Validating;

has @.elements;

	sub plus( $a, $b ) {
		return TensorFlow::Graph.new(
			lhs       => $a,
			operation => '+',
			rhs       => $b
		)
	}

	multi sub infix:<+>( TensorFlow::Tensor $a, Real $b ) is export {
		if $a.elements {
			return TensorFlow::Tensor.new(
				elements => map { $_ + $b }, $a.elements
			);
		}
#die;
		return plus( $a, $b );
	}

	multi sub infix:<+>( Real $a, TensorFlow::Tensor $b ) is export {
		if $b.elements {
			return TensorFlow::Tensor.new(
				elements => map { $_ + $a }, $b.elements
			);
		}
#die;
		return plus( $a, $b );
	}


	multi sub infix:<+>(
		TensorFlow::Tensor $a, TensorFlow::Graph $b ) is export {
		return plus( $a, $b );
	}

	multi sub infix:<+>(
		TensorFlow::Graph $a, TensorFlow::Tensor $b ) is export {
		return plus( $a, $b );
	}

	sub minus( $a, $b ) {
		return TensorFlow::Graph.new(
			lhs       => $a,
			operation => '-',
			rhs       => $b
		)
	}

	multi sub infix:<->( TensorFlow::Tensor $a, Real $b ) is export {
		if $a.elements {
			return TensorFlow::Tensor.new(
				elements => map { $_ - $b }, $a.elements
			);
		}
#die;
		return minus( $a, $b );
	}

	multi sub infix:<->( Real $a, TensorFlow::Tensor $b ) is export {
		if $b.elements {
			return TensorFlow::Tensor.new(
				elements => map { $_ - $a }, $b.elements
			);
		}
#die;
		return minus( $a, $b );
	}

	multi sub infix:<->(
		TensorFlow::Tensor $a, TensorFlow::Graph $b ) is export {
		return minus( $a, $b );
	}

	multi sub infix:<->(
		TensorFlow::Graph $a, TensorFlow::Tensor $b ) is export {
		return minus( $a, $b );
	}

	sub times( $a, $b ) {
		return TensorFlow::Graph.new(
			lhs       => $a,
			operation => '*',
			rhs       => $b
		)
	}

	multi sub infix:<*>( TensorFlow::Tensor $a, Real $b ) is export {
		if $a.elements {
			return TensorFlow::Tensor.new(
				elements => map { $_ * $b }, $a.elements
			);
		}
#die;
		return times( $a, $b );
	}

	multi sub infix:<*>( Real $a, TensorFlow::Tensor $b ) is export {
		if $b.elements {
			return TensorFlow::Tensor.new(
				elements => map { $_ * $a }, $b.elements
			);
		}
#die;
		return times( $a, $b );
	}

	multi sub infix:<*>(
		TensorFlow::Tensor $a, TensorFlow::Graph $b ) is export {
		return times( $a, $b );
	}

	multi sub infix:<*>(
		TensorFlow::Graph $a, TensorFlow::Tensor $b ) is export {
		return times( $a, $b );
	}

	sub divide( $a, $b ) {
		return TensorFlow::Graph.new(
			lhs       => $a,
			operation => '/',
			rhs       => $b
		)
	}

	multi sub infix:</>( TensorFlow::Tensor $a, Real $b ) is export {
		if $a.elements {
			return TensorFlow::Tensor.new(
				elements => map { $_ / $b }, $a.elements
			);
		}
#die;
		return divide( $a, $b );
	}

	multi sub infix:</>( Real $a, TensorFlow::Tensor $b ) is export {
		if $b.elements {
			return TensorFlow::Tensor.new(
				elements => map { $_ / $a }, $b.elements
			);
		}
#die;
		return divide( $a, $b );
	}

	multi sub infix:</>(
		TensorFlow::Tensor $a, TensorFlow::Graph $b ) is export {
		return divide( $a, $b );
	}

	multi sub infix:</>(
		TensorFlow::Graph $a, TensorFlow::Tensor $b ) is export {
		return divide( $a, $b );
	}
}
