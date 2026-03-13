secret_key()

find_max_len_of_dangling_nodes_of_all_pt_files()
#1) search all graphs
#2) store longest chain

find_which_pt_files_should_be_watermarked_from_secret_key()
#1) which files from secret key

select_nodes_within_pt_file_to_be_injected()
#1) identify where it should be injected

inject_pattern_from_longest_chain_on_selected_node_plus_one()
#1) inject rare pattern (longest chain +1)
    #subtask: create randomizer from secret key

rename_label_on_selected_pt_file()

#5) return watermark specifications