=begin pod

=begin NAME

TensorFlow - Bind TensorFlow to Perl 6

=end NAME

=begin SYNOPSIS

    # Passive TensorFlow
    #
    my $v0 = TensorFlow::Tensor.new( 1, 2 );
    $v0 += 2;
    is $v0.[1], 3;

    # Active TensorFlow
    #
    my $v0 = TensorFlow::Tensor.new;
    my $v1 = TensorFlow::Tensor.new( 0, 0 );
    $v1 += $v0;
    TensorFlow.run( $v0, [ TensorFlow::Tensor.new( 1, 2 ) ] );
    is $v1.[0], 1;
    is $v1.[1], 2;

=end SYNOPSIS

=begin DESCRIPTION

TL;DR version - TensorFlow for Perl 6.

=end DESCRIPTION

=begin METHODS

=end METHODS

=end pod

use Inline::Python;

my role Debugging {
}

my role Testing {

}

my role Validating {

}

#use string:from<Python>;
#say string::capwords('foo bar');

#use numpy:from<Python>;
#use tensorflow:from<Python>;

class TensorFlow {
	also does Debugging;
	also does Testing;
	also does Validating;

	has $.ip = Inline::Python.new;

	method float32 {
		$.ip.invoke(
			'__main__', 'tf_wrapper', '_float32'
		);
	}

	method Variable( $value, $type ) {
		$.ip.invoke(
			'__main__', 'tf_wrapper', 'Variable',
			value => $value,
			type => $type
		);
	}

	method test {
		$.ip.run( q:to[_END_] );
import numpy as np
import tensorflow as tf

class tf_wrapper:
	@staticmethod
	def _float32():
		tf.float32

	@staticmethod
	def Variable( value, type ):
		return tf.Variable( value, type )

	@staticmethod
	def run(xxx,yyy,zzz):
		# Model parameters
		#
		#W = tf.Variable([.3], dtype=tf.float32)
		#W = tf.Variable(xxx, dtype=tf.float32)
		W = tf.Variable(xxx, dtype=zzz)
		#b = tf.Variable([-.3], dtype=tf.float32)
		#b = tf.Variable(yyy, dtype=tf.float32)
		b = tf.Variable(yyy, dtype=zzz)

		# Model input and output
		#
		x = tf.placeholder(tf.float32)
		linear_model = W * x + b
		y = tf.placeholder(tf.float32)

		# loss
		#
		# sum of the squares
		#
		loss = tf.reduce_sum(tf.square(linear_model - y))

		# optimizer
		#
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		train = optimizer.minimize(loss)

		# training data
		#
		x_train = [1,2,3,4]
		y_train = [0,-1,-2,-3]

		# training loop
		#
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)

		# reset values to wrong
		#
		for i in range(1000):
		  sess.run(train, {x:x_train, y:y_train})

		# evaluate training accuracy
		#
		curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
		print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
		#return curr_W, curr_b, curr_loss
_END_

#$ip.run('tf_wrapper.run()');
#$ip.call( '__main__', 'tf_wrapper', 'run' );
#$ip.invoke( '__main__', 'tf_wrapper', 'run' );
#$ip.call( '__main__', 'tf_wrapper', 'run', xxx => 0.3 );
#$ip.invoke( '__main__', 'tf_wrapper', 'run', xxx => 0.3 );

		my $float32 = self.float32();

#$ip.invoke( '__main__', 'tf_wrapper', 'run', xxx => 0.3, yyy => -0.3 );
#$ip.invoke( '__main__', 'tf_wrapper', 'run', xxx => [ 0.3 ], yyy => [ -0.3 ] );

		#W = tf.Variable([.3], dtype=tf.float32)

		my $xxx = self.Variable( [ 0.3 ], $float32 );
		$.ip.invoke( '__main__', 'tf_wrapper', 'run',
			xxx => [ 0.3 ],
			yyy => [ -0.3 ],
			zzz => $float32
		);
	}

	method run( $graph, $reference-variables ) {
die;
	}
}
