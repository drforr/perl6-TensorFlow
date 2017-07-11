use v6;

use Test;
use TensorFlow;
use TensorFlow::Tensor;
use TensorFlow::Graph;

use Inline::Python;

my $tf = TensorFlow.new;
my $ip = Inline::Python.new;

#$tf.test();
$tf.rebuild;

#`[

subtest {
	my $v = TensorFlow::Tensor.new;
	is-deeply $v + 2, TensorFlow::Graph.new(
		lhs => $v,
		operation => '+',
		rhs => 2
	);
}, 'graph';

subtest {
	my $v = TensorFlow::Tensor.new(
		elements => [ 1, 2 ]
	);
	is-deeply $v + 2, TensorFlow::Tensor.new(
		elements => [ 3, 4 ]
	);
}, 'regular';

subtest {
	my $v = TensorFlow::Tensor.new;
	my $w = TensorFlow::Tensor.new;
	is-deeply $v * ( $w + 2 ), TensorFlow::Graph.new(
		lhs => TensorFlow::Tensor.new,
		operation => '*',
		rhs => TensorFlow::Graph.new(
			lhs => TensorFlow::Tensor.new,
			operation => '+',
			rhs => 2
		)
	);
}, 'graph';

]

done-testing;

# vim: ft=perl6
