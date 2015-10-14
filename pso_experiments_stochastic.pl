sub modify {
    my $tag = $_[0];
    my $newtag = $_[1];
    my $file = $_[2];
    my $command = 'sed -i \'/'.$tag.'/c\\'.$newtag.'\' '.$file;
    system($command);
}

sub time_since_epoch { return `date +%s` }


$fitnes_function = "stochastic";
$run_dir = "examples/${fitnes_function}/"; 
$config_file = $run_dir."configuration_script.py";
$fitness_file = $run_dir."fitness_script.py";

$sampling = "no";
$m = 100;
$results_folder = "/data/mk306/join/pso_${fitnes_function}_params_${sampling}_sample_m_${m}";
system("rm -Rf $results_folder");
$parallelism = 9;
system("mkdir ".$results_folder); 
$pid = 1;
for ($i=0; $i < $parallelism - 1; $i++) {
    if ($pid) {
        $pid = fork();
    }
    else { 
        last;
    }
}
if ($pid) {
    $i = $parallelism;
}
sleep($i * 10);
$epoch = time_since_epoch();
chomp($epoch);
$tempfile = "/data/mk306/temp_pso_${fitnes_function}_${epoch}_${sampling}_${m}.txt";
$tempfolder = "/data/mk306/extra2_pso_${fitnes_function}_${sampling}_${m}_${epoch}";
system('mkdir '.$tempfolder); 
system("cp -Rf * $tempfolder");
chdir $tempfolder;
system("echo `pwd`");
##modify the configuration file to pso with trails_count
modify('results_folder_path =', 'results_folder_path ="'.$results_folder.'"', $config_file);
modify("trials_type =", 'trials_type =\"PSOTrial\"', $config_file);
modify("surrogate_type =", 'surrogate_type ="proper"', $config_file);
modify("trials_count =", "trials_count = 10", $config_file);
modify("sample_on =", 'sample_on = "no"', $config_file);
modify('corr =', 'corr = "anisotropic"', $config_file);
modify("population_size =", "population_size = 30", $config_file);
modify("M =", "M = 100", $config_file);
modify('goal =', 'goal = "min"', $config_file);

$j = 0;

$kernel_tag  = 'corr ='; 
@kernel = ('corr = "anisotropic"', 'corr = "isotropic"', 'corr = "matern3"'); 
foreach (@kernel) {
    modify($kernel_tag, $_, $config_file);
    $max_stdv_tag  = 'max_stdv =';
    $min_stdv_tag  = 'min_stdv =';
    #@stdv = ('max_stdv = 0.01', 'min_stdv = 0.01'); 
    @stdv = ('max_stdv = 0.01', 'min_stdv = 0.01', 'max_stdv = 0.1', 'min_stdv = 0.1','max_stdv = 0.05', 'min_stdv = 0.05'); 
    foreach (@tuple = splice(@stdv,0,2); @tuple; @tuple = splice(@stdv,0,2)) {
        my ($max_stdv, $min_stdv) = @tuple;
        modify($max_stdv_tag, $max_stdv, $config_file);
        modify($min_stdv_tag, $min_stdv, $config_file);
        if (($j % $parallelism) == ($i - 1)){ #easiest way to divide tasks
            #print $i."\n";
            print "start $j $max_stdv \n";
            my $command = 'make run_example8 > '.$tempfile.' 2>&1';
            system($command);
            print "done $j\n";
        }
        $j = $j + 1;
    }
}
