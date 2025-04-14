using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
public class MainMenuEvents : MonoBehaviour
{
	private UIDocument _document;
	private Button _button;
	public KeyCode Exit = KeyCode.Escape;
	private Button _saveConfigButton;
	public FluidSimulation fluidSimulation;

	private void Start()
	{
		_document = GetComponent<UIDocument>();

		if (_document == null)
		{
			Debug.LogError("UIDocument component missing!");
			return;
		}

		// Initialize buttons with null checks
		_button = _document.rootVisualElement.Q("EnterButton") as Button;
		if (_button != null)
		{
			_button.RegisterCallback<ClickEvent>(OnEnterButtonClick);
		}
		else
		{
			Debug.LogError("EnterButton not found in UI Document!");
		}

		_button = _document.rootVisualElement.Q("QuitButton") as Button;
		if (_button != null)
		{
			_button.RegisterCallback<ClickEvent>(OnQuitButtonClick);
		}
		else
		{
			Debug.LogError("QuitButton not found in UI Document!");
		}

		_saveConfigButton = _document.rootVisualElement.Q("SaveConfigButton") as Button;
		if (_saveConfigButton != null)
		{
			_saveConfigButton.RegisterCallback<ClickEvent>(OnSaveConfigButtonClick);
		}
		else
		{
			Debug.LogError("SaveConfigButton not found in UI Document!");
		}
	}
	private void Update()
	{
		if (Input.GetKeyDown(Exit))
		{
			bool wasVisible = _document.rootVisualElement.style.display == DisplayStyle.Flex;
			_document.rootVisualElement.style.display = wasVisible ? DisplayStyle.None : DisplayStyle.Flex;
		}
	}

	private void OnEnterButtonClick(ClickEvent evt)
	{
		_document.rootVisualElement.style.display = DisplayStyle.None;
	}

	private void OnQuitButtonClick(ClickEvent evt)
	{
		// Closes application
		Application.Quit();
		Debug.Log("Application.Quit() called");


		//Temporary stop play mode in Unity editor
#if UNITY_EDITOR
		UnityEditor.EditorApplication.isPlaying = false;
#endif
	}

	private void OnSaveConfigButtonClick(ClickEvent evt)
	{
		if (fluidSimulation == null)
		{
			// Dynamically find the FluidSimulation instance if not assigned
			fluidSimulation = FindObjectOfType<FluidSimulation>();

			if (fluidSimulation == null)
			{
				Debug.LogError("FluidSimulation reference is missing!");
				return;
			}
		}

		fluidSimulation.SaveCurrentConfiguration();
		Debug.Log("Parameters saved to SQL.");
	}
}