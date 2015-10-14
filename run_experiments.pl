$run_dir = "/examples/quadrature_method_based_app/"; 
$config_file = $run_dir."configuration_script.py";
for (my $i=0; $i <= 9; $i++) {
        #modify configuration file
        $tag = 'goal';
        $newtag = "goal = $i";
	print "sed -i '/$tag/c\\$newtag' ".$config_file;
}
