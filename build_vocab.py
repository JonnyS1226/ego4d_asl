import torch
#如果没有torchtext包 需要使用命令 pip install torchtext  安装torchtext包
import torchtext.vocab as vocab
import numpy as np
import pickle
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

# 计算余弦相似度
def Cos(x, y):
    cos = torch.matmul(x, y.view((-1,))) / (
            (torch.sum(x * x) + 1e-9).sqrt() * torch.sum(y * y).sqrt())
    return cos
 
if __name__ == '__main__':
    total = np.array([])

    word_emb_type = 'clip'
    #选择自己所需要的词向量集
    if word_emb_type == 'glove':
        glove = vocab.GloVe(name="6B", dim=300)
    elif word_emb_type == 'clip':
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # No Finding类别  
    # glove.stoi[]方法是获取对应的词向量的index下标
    # glove.vectors[]方法是获取对应词向量下标的词向量
    # no = glove.vectors[glove.stoi['no']]
    # finding = glove.vectors[glove.stoi['finding']]
    # no_finding = no + finding
    # total = np.append(total, no_finding.numpy())
    
    classes = {'use_phone': 16, 'water_soil_/_plants_/_crops': 59, 'clean_/_wipe_a_table_or_kitchen_counter': 29, 'walk_down_stairs_/_walk_up_stairs': 30, 'arrange_/_organize_other_items': 24, 'clean_/_wipe_other_surface_or_object': 6, 'fill_a_pot_/_bottle_/_container_with_water': 64, 'use_a_laptop_/_computer': 37, 'knead_/_shape_/_roll-out_dough': 22, 'cut_dough': 25, 'fry_dough': 57, 'converse_/_interact_with_someone': 11, 'stir_/_mix_food_while_cooking': 21, 'wash_dishes_/_utensils_/_bakeware_etc.': 68, 'turn-on_/_light_the_stove_burner': 9, 'serve_food_onto_a_plate': 67, 'chop_/_cut_wood_pieces_using_tool': 82, 'cut_/_trim_grass_with_other_tools': 92, 'trim_hedges_or_branches': 100, 'browse_through_groceries_or_food_items_on_rack_/_shelf': 32, 'read_a_book_/_magazine_/_shopping_list_etc.': 28, 'take_photo_/_record_video_with_a_camera': 0, 'pay_at_billing_counter': 42, 'stand_in_the_queue_/_line_at_a_shop_/_supermarket': 53, 'browse_through_other_items_on_rack_/_shelf': 50, 'browse_through_clothing_items_on_rack_/_shelf_/_hanger': 2, 'look_at_clothes_in_the_mirror': 83, '"try-out_/_wear_accessories_(e.g._tie,_belt,_scarf)"': 102, 'put_away_(or_take_out)_dishes_/_utensils_in_storage': 81, 'clean_/_wipe_kitchen_appliance': 23, 'wash_vegetable_/_fruit_/_food_item': 95, '"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat"': 75, 'cut_other_item_using_tool': 27, 'drill_into_wall_/_wood_/_floor_/_metal': 19, 'use_hammer_/_nail-gun_to_fix_nail': 34, 'weigh_food_/_ingredient_using_a_weighing_scale': 54, 'pack_food_items_/_groceries_into_bags_/_boxes': 41, 'drink_beverage': 65, 'withdraw_money_from_atm_/_operate_atm': 3, 'put_away_(or_take_out)_food_items_in_the_fridge': 39, 'interact_or_play_with_pet_/_animal': 101, 'put_away_(or_take_out)_ingredients_in_storage': 7, '"try-out_/_wear_clothing_items_(e.g._shirt,_jeans,_sweater)"': 77, 'throw_away_trash_/_put_trash_in_trash_can': 8, 'tie_up_branches_/_plants_with_string': 103, 'remove_weeds_from_ground': 85, 'collect_/_rake_dry_leaves_on_ground': 91, 'harvest_vegetables_/_fruits_/_crops_from_plants_on_the_ground': 86, 'place_items_in_shopping_cart': 31, 'write_notes_in_a_paper_/_book': 108, 'wash_hands': 5, 'pack_other_items_into_bags_/_boxes': 73, 'pack_soil_into_the_ground_or_a_pot_/_container': 47, 'plant_seeds_/_plants_/_flowers_into_ground': 48, '"level_ground_/_soil_(eg._using_rake,_shovel,_etc)"': 46, 'dig_or_till_the_soil_with_a_hoe_or_other_tool': 45, 'cut_tree_branch': 90, 'measure_wooden_item_using_tape_/_ruler': 35, 'mark_item_with_pencil_/_pen_/_marker': 36, 'compare_two_clothing_items': 97, 'do_some_exercise': 80, 'watch_television': 17, 'taste_food_while_cooking': 96, 'rinse_/_drain_other_food_item_in_sieve_/_colander': 71, 'use_a_vacuum_cleaner_to_clean': 15, 'fix_other_item': 20, 'smooth_wood_using_sandpaper_/_sander_/_tool': 88, 'dig_or_till_the_soil_by_hand': 98, 'hang_clothes_in_closet_/_on_hangers': 1, 'clean_/_wipe_/_oil_metallic_item': 72, 'fix_bonnet_/_engine_of_car': 107, 'hang_clothes_to_dry': 109, 'cut_/_trim_grass_with_a_lawnmower': 76, 'fold_clothes_/_sheets': 56, 'dismantle_other_item': 18, 'fix_/_remove_/_replace_a_tire_or_wheel': 84, 'move_/_shift_/_arrange_small_tools': 78, 'make_coffee_or_tea_/_use_a_coffee_machine': 63, 'play_board_game_or_card_game': 60, 'count_money_before_paying': 40, 'enter_a_supermarket_/_shop': 49, 'exit_a_supermarket_/_shop': 51, 'play_a_video_game': 79, 'arrange_pillows_on_couch_/_chair': 104, '"make_the_bed_/_arrange_pillows,_sheets_etc._on_bed"': 105, 'clean_/_sweep_floor_with_broom': 61, 'arrange_/_organize_clothes_in_closet/dresser': 55, 'load_/_unload_a_washing_machine_or_dryer': 89, 'move_/_shift_around_construction_material': 70, '"put_on_safety_equipment_(e.g._gloves,_helmet,_safety_goggles)"': 52, 'cut_open_a_package_(e.g._with_scissors)': 66, 'stir_/_mix_ingredients_in_a_bowl_or_pan_(before_cooking)': 4, 'fry_other_food_item': 38, 'eat_a_snack': 62, 'drive_a_vehicle': 99, 'arrange_/_organize_items_in_fridge': 10, 'browse_through_accessories_on_rack_/_shelf': 43, 'fix_wiring': 26, 'prepare_or_apply_cement_/_concrete_/_mortar': 69, 'put_food_into_the_oven_to_bake': 106, 'peel_a_fruit_or_vegetable': 74, 'smoke_cigar_/_cigarette_/_vape': 93, 'paint_using_paint_brush_/_roller': 14, 'climb_up_/_down_a_ladder': 12, 'cut_thread_/_paper_/_cardboard_using_scissors_/_knife_/_cutter': 44, 'plaster_wall_/_surface': 13, 'fix_pipe_/_plumbing': 87, '"clean_/_repair_small_equipment_(mower,_trimmer_etc.)"': 33, 'remove_food_from_the_oven': 58, 'iron_clothes_or_sheets': 94}
    classes_list = [''] * len(classes)
    for k, v in classes.items():
        if word_emb_type == "glove":
            classes_list[v] = k.replace('/', ' or ').replace('_', ' ').replace('(', ' ').replace(')', ' ').replace('"', '').replace(',', '').replace('nail-gun','nail gun').replace('e.g.', 'for example').replace('eg.', 'for example')
        else:
            classes_list[v] = k.replace('/', ' or ').replace('_', ' ')
    for class_item in tqdm(classes_list):
        if word_emb_type == 'glove':
            class_tokens = class_item.split(' ')
            feature = glove.vectors[glove.stoi[class_tokens[0]]]
            for token in class_tokens:
                if token == class_tokens[0] or token == '' or token == 'vape':
                    continue
                feature += glove.vectors[glove.stoi[token]]
            total = np.append(total, feature.numpy())
        else:
            inputs = tokenizer(class_item, padding=True, return_tensors="pt")
            outputs = model(**inputs)
            feature = outputs.pooler_output
            total = np.append(total, feature.detach().numpy())
        

    #我总共有14个类别，所以这个地方写14
    total = total.reshape(110, -1)      # [c,d]
    #保存对应类别的word embedding
    pickle.dump(total, open(f'data/ego4d/{word_emb_type}_wordEmbedding.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
 
    # #可以打印输出看一下余弦相似度
    # print("NO Finding和Fracture的余弦相似度：", Cos(no_finding, fracture))
    # print("Lung Opacity和Atelectasis的余弦相似度:", Cos(lung_opacity, atelectasis))
    # print("Lung Opactiy和Fracture的余弦相似度:", Cos(lung_opacity, fracture))
 