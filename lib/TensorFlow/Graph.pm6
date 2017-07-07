my role Debugging {
}

my role Testing {

}

my role Validating {

}

class TensorFlow::Graph {
	also does Debugging;
	also does Testing;
	also does Validating;

	has $.lhs;
	has $.operation;
	has $.rhs;
}
