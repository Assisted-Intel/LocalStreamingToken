#!/usr/bin/env python3
"""The /api/settings allowlist.

The route patches only the keys it names, which is what lets a single control save
itself without round-tripping every other field. That property is easy to break by
adding a key to the UI and forgetting the allowlist — the setting then appears to
save and silently reverts on the next load.
"""


def test_a_one_key_patch_leaves_the_other_settings_alone(client):
    client.post("/api/settings", json={"default_num_ctx": 8192})
    cfg = client.post("/api/settings", json={"max_output_tokens": 4321}
                      ).get_json()["config"]
    assert cfg["max_output_tokens"] == 4321
    assert cfg["default_num_ctx"] == 8192


def test_an_unknown_key_is_ignored_rather_than_stored(client):
    cfg = client.post("/api/settings", json={"not_a_real_setting": "x"}).get_json()["config"]
    assert "not_a_real_setting" not in cfg


def test_the_chat_settings_collapse_state_persists(client):
    """The chat tab's Settings region is collapsed by default and remembers the
    choice globally, so the preference has to survive the round trip."""
    assert client.post("/api/settings", json={"chat_settings_collapsed": True}
                       ).get_json()["config"]["chat_settings_collapsed"] is True
    assert client.get("/api/state").get_json()["config"]["chat_settings_collapsed"] is True
    assert client.post("/api/settings", json={"chat_settings_collapsed": False}
                       ).get_json()["config"]["chat_settings_collapsed"] is False


def test_a_non_boolean_collapse_state_is_coerced(client):
    """The value reaches the client as a class toggle; a stray string would read as
    truthy in Python and as its own truthiness in JS. Store a real bool."""
    cfg = client.post("/api/settings", json={"chat_settings_collapsed": "yes"}).get_json()["config"]
    assert cfg["chat_settings_collapsed"] is True
