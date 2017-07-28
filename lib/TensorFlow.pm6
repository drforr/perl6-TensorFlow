=begin pod

=begin NAME

TensorFlow - Bind TensorFlow to Perl 6

=end NAME

=begin SYNOPSIS

#    # Passive TensorFlow
#    #
#    my $v0 = TensorFlow::Tensor.new( 1, 2 );
#    $v0 += 2;
#    is $v0.[1], 3;
#
#    # Active TensorFlow
#    #
#    my $v0 = TensorFlow::Tensor.new;
#    my $v1 = TensorFlow::Tensor.new( 0, 0 );
#    $v1 += $v0;
#    TensorFlow.run( $v0, [ TensorFlow::Tensor.new( 1, 2 ) ] );
#    is $v1.[0], 1;
#    is $v1.[1], 2;

=end SYNOPSIS

=begin DESCRIPTION

TL;DR version - TensorFlow for Perl 6.

=end DESCRIPTION

=begin METHODS

=end METHODS

=end pod

use Inline::Python;

class TensorFlow::Type {
	has $.python;
	has $.value;
}
class TensorFlow::Value {
	has $.python;
	has $.value;
}
class TensorFlow::Variable {
	also is TensorFlow::Value;
}
class TensorFlow::placeholder {
	also is TensorFlow::Value;
}
class TensorFlow::trainer {
	also is TensorFlow::Value;
}
class TensorFlow::optimizer {
	also is TensorFlow::Value;

	method minimize( $loss ) {
		return TensorFlow::trainer.new( python => $.python )
	}
}
class TensorFlow::train {
	also is TensorFlow::Value;

	method GradientDescentOptimizer( $value ) {
		return TensorFlow::optimizer.new(
			python => $.python,
			value => $.python.call(
				'__main__',
				'GradientDescentOptimizer',
				$value
			)
		)
	}
}
class TensorFlow::initializer {
	also is TensorFlow::Value;
}
class TensorFlow::Session {
	also is TensorFlow::Value;

	multi method run( TensorFlow::trainer $value, $hash ) {
	}
	multi method run( $value, $hash ) {
	}
	multi method run( $value ) {
	}
}

# Notes for the internals:
#
# For Pete's sake don't use __ in method names, and don't use trailing '_'.
# It's reserved for ... things that confuse python.
#
class TensorFlow {
	has $.python;

	my $instance;

	method _wrapper {
		return q:to[_END_];
import numpy as np
import tensorflow as tf

def float32():
	return tf.float32
def Variable(value,type):
	return tf.Variable( value, dtype = type )
def placeholder( type ):
	return tf.placeholder( type )

# It's easier than explicitly calling __add__ etcetera.
def plus( a, b ):
	return a + b
def times( a, b ):
	return a * b
def minus( a, b ):
	return a - b
def div( a, b ):
	return a / b
def mod( a, b ):
	return a % b

def square( value ):
	return tf.square( value )
def reduce_sum( value ):
	return tf.reduce_sum( value )

def GradientDescentOptimizer( value ):
	return tf.train.GradientDescentOptimizer( value )

def global_variables_initializer():
	return tf.global_values_initializer()
_END_
	}

	method new {
		unless $instance {
			my $python = Inline::Python.new;
			$python.run( self._wrapper );
			$instance = self.bless(
				 python => $python
			);
		}
		$instance;
	}

	method float32 {
		return TensorFlow::Type.new(
			python => $.python,
			value => $.python.call('__main__','float32')
		)
	}
	method Variable( $value, :$type ) {
		return TensorFlow::Variable.new(
			python => $.python,
			value => $.python.call(
				'__main__',
				'Variable',
				$value,
				$type
			)
		)
	}
	method placeholder( $type ) {
		return TensorFlow::placeholder.new(
			python => $.python,
			value => $.python.call(
				'__main__',
				'placeholder',
				$type.value
			)
		)
	}
	multi sub infix:<*>( TensorFlow::Variable $a,
			     TensorFlow::placeholder $b ) is export {
		return TensorFlow::Value.new(
			python => $a.python,
			value => $a.python.call(
				'__main__',
				'times',
				$a.value,
				$b.value
			)
		);
	}
	multi sub infix:<+>( TensorFlow::Value $a,
			     TensorFlow::Variable $b ) is export {
		return TensorFlow::Value.new(
			python => $a.python,
			value => $a.python.call(
				'__main__',
				'plus',
				$a.value,
				$b.value
			)
		)
	}
	multi sub infix:<->( TensorFlow::Value $a,
			     TensorFlow::placeholder $b ) is export {
		return TensorFlow::Value.new(
			python => $a.python,
			value => $a.python.call(
				'__main__',
				'minus',
				$a.value,
				$b.value
			)
		)
	}
	method square( $value ) {
		return TensorFlow::Value.new(
			python => $value.python,
			value => $value.python.call(
				'__main__',
				'square',
				$value.value
			)
		)
	}
	method reduce_sum( $value ) {
		return TensorFlow::Value.new(
			python => $value.python,
			value => $value.python.call(
				'__main__',
				'reduce_sum',
				$value.value
			)
		)
	}
	method train {
		return TensorFlow::train.new( python => $.python )
	}
	method global_variables_initializer {
		return TensorFlow::initializer.new(
			python => $.python,
#			value => $.python.call(
#				'__main__',
#				'global_variables_initializer'
#			)
		)
	}
	method Session {
		return TensorFlow::Session.new( python => $.python )
	}
}
