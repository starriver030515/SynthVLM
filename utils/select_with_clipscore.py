import json

def select_with_clipscore(args):
    with open(args.clipscore_path) as f:
        clipscore = json.load(f)
        
    filtered = []
    for i in range(0, len(clipscore), args.repeat):
        group = clipscore[i:i + args.repeat]
        top_image = max(group, key=lambda x: x['clipscore'])
        filtered.append(top_image)
        
    number_to_keep = int(len(filtered) * args.select_radio)
    sorted_filtered = sorted(filtered, key=lambda x: x['clipscore'], reverse=True)
    sorted_filtered = sorted_filtered[:number_to_keep]
    
    with open(args.filtered_clipscore_path, 'w') as f:
        json.dump(sorted_filtered, f, indent=4)