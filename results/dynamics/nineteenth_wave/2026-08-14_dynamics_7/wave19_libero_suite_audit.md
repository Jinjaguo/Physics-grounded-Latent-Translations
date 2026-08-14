# Wave-19 LIBERO suite audit

```text
requested_internal_name = LIBERO-Long
resolved_official_suite_name = libero_10
number_of_tasks = 10
exact_task_list = [
  0: put both the alphabet soup and the tomato sauce in the basket
  1: put both the cream cheese box and the butter in the basket
  2: turn on the stove and put the moka pot on it
  3: put the black bowl in the bottom drawer of the cabinet and close it
  4: put the white mug on the left plate and put the yellow and white mug on the right plate
  5: pick up the book and place it in the back compartment of the caddy
  6: put the white mug on the plate and put the chocolate pudding to the right of the plate
  7: put both the alphabet soup and the cream cheese box in the basket
  8: put both moka pots on the stove
  9: put the yellow and white mug in the microwave and close it
]
```

Available installed suites: ['libero_10', 'libero_100', 'libero_90', 'libero_goal', 'libero_object', 'libero_spatial']

## Task 00

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl`
- language: put both the alphabet soup and the tomato sauce in the basket
- success predicate: `(:goal
    (And (In alphabet_soup_1 basket_1_contain_region) (In tomato_sauce_1 basket_1_contain_region))
  )`
- controller: `OSC_POSE`

## Task 01

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl`
- language: put both the cream cheese box and the butter in the basket
- success predicate: `(:goal
    (And (In cream_cheese_1 basket_1_contain_region) (In butter_1 basket_1_contain_region))
  )`
- controller: `OSC_POSE`

## Task 02

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl`
- language: turn on the stove and put the moka pot on it
- success predicate: `(:goal
    (And (Turnon flat_stove_1) (On moka_pot_1 flat_stove_1_cook_region))
  )`
- controller: `OSC_POSE`

## Task 03

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl`
- language: put the black bowl in the bottom drawer of the cabinet and close it
- success predicate: `(:goal
    (And (Close white_cabinet_1_bottom_region) (In akita_black_bowl_1 white_cabinet_1_bottom_region))
  )`
- controller: `OSC_POSE`

## Task 04

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl`
- language: put the white mug on the left plate and put the yellow and white mug on the right plate
- success predicate: `(:goal
    (And (On porcelain_mug_1 plate_1) (On white_yellow_mug_1 plate_2))
  )`
- controller: `OSC_POSE`

## Task 05

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl`
- language: pick up the book and place it in the back compartment of the caddy
- success predicate: `(:goal
    (And (In black_book_1 desk_caddy_1_back_contain_region))
  )`
- controller: `OSC_POSE`

## Task 06

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl`
- language: put the white mug on the plate and put the chocolate pudding to the right of the plate
- success predicate: `(:goal
    (And (On porcelain_mug_1 plate_1) (On chocolate_pudding_1 living_room_table_plate_right_region))
  )`
- controller: `OSC_POSE`

## Task 07

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl`
- language: put both the alphabet soup and the cream cheese box in the basket
- success predicate: `(:goal
    (And (In alphabet_soup_1 basket_1_contain_region) (In cream_cheese_1 basket_1_contain_region))
  )`
- controller: `OSC_POSE`

## Task 08

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl`
- language: put both moka pots on the stove
- success predicate: `(:goal
    (And (On moka_pot_1 flat_stove_1_cook_region) (On moka_pot_2 flat_stove_1_cook_region) (Turnon flat_stove_1))
  )`
- controller: `OSC_POSE`

## Task 09

- BDDL: `/home/jinjaguo/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl`
- language: put the yellow and white mug in the microwave and close it
- success predicate: `(:goal
    (And (In white_yellow_mug_1 microwave_1_heating_region) (Close microwave_1))
  )`
- controller: `OSC_POSE`
